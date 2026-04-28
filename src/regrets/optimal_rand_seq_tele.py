from api.client import get_client
from openai import AssistantEventHandler, OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
import re
import tiktoken
from transformers import AutoTokenizer
import time


# Pre‐instantiate a fallback tokenizer (BERT) to use if LLaMA loading fails:
_fallback_tok = AutoTokenizer.from_pretrained("bert-base-uncased")


def opt_eval(deployments, prompt, task, selected, avg_array, t,
             all_rewards_sum, all_rewards_diag, labels, dataset):
    """
    For telecom:
      • If task == "diagnosis": run each arm on `prompt` and match “option \\d+” vs. labels[t]
      • If task == "explanation": run each arm on `prompt` to generate an explanation,
        compare generated text to labels[t] (gold‐explanation), assign reward (1/0).

    Returns: (regret, reward, out_len, updated_avg_array, updated_all_rewards_sum, updated_all_rewards_diag)
    """
    result = {}
    actual = ""
    arm_to_llm = {
        "base"            : "gpt-3.5-turbo",
        "assistants"      : "gpt-3.5-turbo",
        "finetune_med"    : "gpt-4",
        "finetune_tele"   : "gpt-4",
        "finetune_med_new": "gpt-4",
        "llama"           : "llama-13b"
    }

    reward  = None
    out_len = 0

    # ─── NEW BRANCH FOR EXPLANATION SUBTASK ────────────────────────────────────────
    if task == "explanation":
        gold_explanation = labels[t]

        for cat in deployments:
            match = None

            if cat == "finetune_med":
                actual     = "Med"
                cat_lookup = "finetune"
            elif cat == "finetune_med_new":
                actual     = "Med_New"
                cat_lookup = "finetune"
            elif cat == "finetune_tele":
                actual     = "Tele"
                cat_lookup = "finetune"
            else:
                cat_lookup = cat

            client = get_client(cat_lookup)
            while match is None:
                # “assistants”‐style (Azure + file_search):
                if cat == "assistants":
                    for x in deployments[cat]:
                        name, context = x
                        assistant = client.beta.assistants.create(
                            name="Explanation Generator",
                            instructions=context,
                            tools=[{"type": "file_search"}],
                            model=name
                        )
                        vector_store = client.beta.vector_stores.create(name="ExplanationDocs")

                        # Write prompt to disk so “assistants” tool can search it:
                        with open("data/input.txt", "w", encoding="utf-8") as f:
                            f.write(prompt)

                        file_paths   = ["data/input.txt"]
                        file_streams = [open(path, "rb") for path in file_paths]
                        file_batch   = client.beta.vector_stores.file_batches.upload_and_poll(
                            vector_store_id=vector_store.id, files=file_streams
                        )

                        assistant = client.beta.assistants.update(
                            assistant_id=assistant.id,
                            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
                        )

                        message_file = client.files.create(
                            file=open("data/input.txt", "rb"), purpose="assistants"
                        )
                        thread = client.beta.threads.create(
                            messages=[{
                                "role": "user",
                                "content": prompt,
                                "attachments": [
                                    {
                                        "file_id": message_file.id,
                                        "tools": [{"type": "file_search"}]
                                    }
                                ],
                            }]
                        )

                        class EventHandler(AssistantEventHandler):
                            def on_message_done(self, message) -> None:
                                msg_content = message.content[0].text
                                with open("data/outputs.txt", "w", encoding="utf-8") as fo:
                                    fo.write(msg_content)

                        with client.beta.threads.runs.stream(
                            thread_id=thread.id,
                            assistant_id=assistant.id,
                            instructions="Please address the user as Jane Doe. The user has a premium account.",
                            event_handler=EventHandler()
                        ) as stream:
                            stream.until_done()

                        with open("data/outputs.txt", "r", encoding="utf-8") as fo:
                            generated = fo.read()

                        # If this was the selected arm, compute out_len:
                        if cat == selected:
                            openai_models = {"gpt-3.5-turbo", "gpt-4"}
                            encodings     = {m: tiktoken.encoding_for_model(m) for m in openai_models}

                            # Try loading LLaMA tokenizer; if it fails, fall back to BERT:
                            try:
                                llm_llama_tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
                            except Exception:
                                llm_llama_tok = _fallback_tok

                            llm_name = arm_to_llm[selected]
                            text     = generated if isinstance(generated, str) else str(generated)
                            if llm_name in encodings:
                                out_len = len(encodings[llm_name].encode(text))
                            else:
                                out_len = len(llm_llama_tok(text, truncation=True, padding=False)["input_ids"])

                        match = generated  # exit the loop for this arm
                        message = generated

                # “small / llama / deepseek / phi” (non‐Azure):
                elif cat in ["small", "llama", "deepseek", "phi"]:
                    name, context = deployments[cat][0], deployments[cat][1]
                    response = client.complete(
                        messages=[SystemMessage(content=context), UserMessage(content=prompt)],
                        max_tokens=2048,
                        temperature=0.8,
                        top_p=0.1,
                        model=name
                    )
                    message = response.choices[0].message.content

                    if cat == selected:
                        openai_models = {"gpt-3.5-turbo", "gpt-4"}
                        encodings     = {m: tiktoken.encoding_for_model(m) for m in openai_models}

                        try:
                            llm_llama_tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
                        except Exception:
                            llm_llama_tok = _fallback_tok

                        llm_name = arm_to_llm[selected]
                        text     = message if isinstance(message, str) else str(message)
                        if llm_name in encodings:
                            out_len = len(encodings[llm_name].encode(text))
                        else:
                            out_len = len(llm_llama_tok(text, truncation=True, padding=False)["input_ids"])

                    match = message

                # “finetune” (Azure OpenAI) path:
                else:
                    if actual == "Med":
                        cat_lookup = "finetune_med"
                    elif actual == "Tele":
                        cat_lookup = "finetune_tele"
                    elif actual == "Med_New":
                        cat_lookup = "finetune_med_new"

                    name, context = deployments[cat_lookup][0], deployments[cat_lookup][1]
                    response = client.chat.completions.create(
                        model=name,
                        messages=[{"role": "system", "content": context},
                                  {"role": "user",   "content": prompt}]
                    )
                    message = response.choices[0].message.content

                    if cat == selected:
                        openai_models = {"gpt-3.5-turbo", "gpt-4"}
                        encodings     = {m: tiktoken.encoding_for_model(m) for m in openai_models}

                        try:
                            llm_llama_tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
                        except Exception:
                            llm_llama_tok = _fallback_tok

                        llm_name = arm_to_llm[selected]
                        text     = message if isinstance(message, str) else str(message)
                        if llm_name in encodings:
                            out_len = len(encodings[llm_name].encode(text))
                        else:
                            out_len = len(llm_llama_tok(text, truncation=True, padding=False)["input_ids"])

                    match = message

            from difflib import SequenceMatcher

            # normalize both strings to lowercase
            gen = message.lower() if isinstance(message, str) else str(message).lower()
            gold = gold_explanation.lower()

            # SequenceMatcher.ratio() returns a float in [0,1]
            score = SequenceMatcher(None, gold, gen).ratio()

            all_rewards_diag.append(score)
            avg_array[name] = (avg_array[name] * t + score) / (t + 1)
            result[name]    = score

            if cat == selected or actual == selected:
                reward = score

        # Now pick the best score among all arms in result:
        best_score = 0
        best_dep   = None
        for dep, sc in result.items():
            if sc >= best_score:
                best_score = sc
                best_dep   = dep

        if reward is None:
            raise ValueError("Reward was not calculated for the selected explainer deployment.")

        regret = best_score - reward
        return regret, reward, out_len, avg_array, all_rewards_sum, all_rewards_diag

    # ─── FALL BACK TO THE ORIGINAL “DIAGNOSIS” BRANCH ──────────────────────────────
    for cat in deployments:
        match = None

        if cat == "finetune_med":
            cat_lookup = "finetune"
            actual     = "Med"
        elif cat == "finetune_med_new":
            cat_lookup = "finetune"
            actual     = "Med_New"
        elif cat == "finetune_tele":
            cat_lookup = "finetune"
            actual     = "Tele"
        else:
            cat_lookup = cat

        client = get_client(cat_lookup)

        while match is None:
            if cat == "assistants":
                for x in deployments[cat]:
                    name, context = x
                    assistant = client.beta.assistants.create(
                        name="Diagnosis Summarizer",
                        instructions=context,
                        tools=[{"type": "file_search"}],
                        model=name
                    )
                    vector_store = client.beta.vector_stores.create(name="Diagnosis Reports")

                    with open("data/input.txt", "w", encoding="utf-8") as f:
                        f.write(prompt)

                    file_paths   = ["data/input.txt"]
                    file_streams = [open(path, "rb") for path in file_paths]
                    file_batch   = client.beta.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store.id, files=file_streams
                    )

                    assistant = client.beta.assistants.update(
                        assistant_id=assistant.id,
                        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
                    )

                    message_file = client.files.create(
                        file=open("data/input.txt", "rb"), purpose="assistants"
                    )
                    thread = client.beta.threads.create(
                        messages=[{
                            "role": "user",
                            "content": prompt,
                            "attachments": [
                                {"file_id": message_file.id, "tools": [{"type": "file_search"}]}
                            ],
                        }]
                    )

                    class EventHandler(AssistantEventHandler):
                        def on_message_done(self, message) -> None:
                            message_content = message.content[0].text
                            annotations     = message_content.annotations
                            for index, annotation in enumerate(annotations):
                                message_content.value = message_content.value.replace(
                                    annotation.text, f"[{index}]"
                                )
                                if file_citation := getattr(annotation, "file_citation", None):
                                    cited_file = client.files.retrieve(file_citation.file_id)
                                    # citations not used further here

                            with open("data/outputs.txt", "w", encoding="utf-8") as fo:
                                fo.write(message_content.value)

                    with client.beta.threads.runs.stream(
                        thread_id=thread.id,
                        assistant_id=assistant.id,
                        instructions="Please address the user as Jane Doe. The user has a premium account.",
                        event_handler=EventHandler(),
                    ) as stream:
                        stream.until_done()

                    with open("data/outputs.txt", "r", encoding="utf-8") as d:
                        message = d.read()

                    if cat == selected:
                        openai_models = {"gpt-3.5-turbo", "gpt-4"}
                        encodings     = {m: tiktoken.encoding_for_model(m) for m in openai_models}

                        try:
                            llm_llama_tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
                        except Exception:
                            llm_llama_tok = _fallback_tok

                        llm_name = arm_to_llm[selected]
                        text     = message if isinstance(message, str) else str(message)
                        if llm_name in encodings:
                            out_len = len(encodings[llm_name].encode(text))
                        else:
                            out_len = len(llm_llama_tok(text, truncation=True, padding=False)["input_ids"])

                    if dataset == "telecom":
                        match = re.search(r"option \s*(\d+)", message)
                        label = re.search(r"option \s*(\d+)", labels[t])
                        if match is None:
                            match = re.search(r"Option \s*(\d+)", message)
                        if match:
                            num = int(match.group(1))
                            lab = int(label.group(1))
                            score = 1 if (lab == num) else 0
                            all_rewards_diag.append(score)
                            avg_array[name] = (avg_array[name] * t + score) / (t + 1)
                            result[name]    = score
                            if cat == selected or actual == selected:
                                reward = score

                    elif dataset == "medical":
                        cnt = 0
                        if message is None:
                            message = ""
                        elif not isinstance(message, str):
                            if hasattr(message, "content"):
                                message = message.content
                            elif hasattr(message, "text"):
                                message = message.text
                            else:
                                message = str(message)
                        new_msg = message.lower()
                        for el in labels[t]:
                            if new_msg.find(el) != -1:
                                cnt += 1

                        score = cnt / len(labels[t])
                        all_rewards_diag.append(score)
                        avg_array[name] = (avg_array[name] * t + score) / (t + 1)
                        result[name]    = score
                        if cat == selected or actual == selected:
                            reward = score

                    match = match  # break out of while

            elif cat in ["small", "llama", "deepseek", "phi"]:
                name, context = deployments[cat][0], deployments[cat][1]
                response = client.complete(
                    messages=[SystemMessage(content=context), UserMessage(content=prompt)],
                    max_tokens=2048,
                    temperature=0.8,
                    top_p=0.1,
                    model=name
                )
                message = response.choices[0].message.content

                if cat == selected:
                    openai_models = {"gpt-3.5-turbo", "gpt-4"}
                    encodings     = {m: tiktoken.encoding_for_model(m) for m in openai_models}

                    try:
                        llm_llama_tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
                    except Exception:
                        llm_llama_tok = _fallback_tok

                    llm_name = arm_to_llm[selected]
                    text     = message if isinstance(message, str) else str(message)
                    if llm_name in encodings:
                        out_len = len(encodings[llm_name].encode(text))
                    else:
                        out_len = len(llm_llama_tok(text, truncation=True, padding=False)["input_ids"])

                if dataset == "telecom":
                    if message == "{5}":
                        message = "option 5"
                    if message.isnumeric():
                        message = f"option {message}"
                    if "{" in message or "}" in message:
                        message = message.replace("{","").replace("}","")
                        message = f"option {message}"

                    for num in ["1", "2", "3", "4", "5"]:
                        if message.startswith(num):
                            message = f"option {num}"
                        elif message.find(f"{num}:") != -1 or message.find(f"{num}: ") != -1:
                            message = f"option {num}"
                        elif message.splitlines()[-1].startswith(num):
                            message = f"option {num}"

                    match = re.search(r"option \s*(\d+)", message)
                    label = re.search(r"option \s*(\d+)", labels[t])
                    if match is None:
                        match = re.search(r"Option \s*(\d+)", message)
                    if match:
                        num = int(match.group(1))
                        lab = int(label.group(1))
                        score = 1 if (lab == num) else 0
                        all_rewards_diag.append(score)
                        avg_array[name] = (avg_array[name] * t + score) / (t + 1)
                        result[name]    = score
                        if cat == selected or actual == selected:
                            reward = score

                elif dataset == "medical":
                    cnt = 0
                    if message is None:
                        message = ""
                    elif not isinstance(message, str):
                        if hasattr(message, "content"):
                            message = message.content
                        elif hasattr(message, "text"):
                            message = message.text
                        else:
                            message = str(message)
                    new_msg = message.lower()
                    for el in labels[t]:
                        if new_msg.find(el) != -1:
                            cnt += 1
                    score = cnt / len(labels[t])
                    all_rewards_diag.append(score)
                    avg_array[name] = (avg_array[name] * t + score) / (t + 1)
                    result[name]    = score
                    if cat == selected or actual == selected:
                        reward = score

                match = match  # exit while

            else:
                # “finetune” (Azure OpenAI) path
                if actual == "Med":
                    cat_lookup = "finetune_med"
                elif actual == "Tele":
                    cat_lookup = "finetune_tele"
                elif actual == "Med_New":
                    cat_lookup = "finetune_med_new"
                name, context = deployments[cat_lookup][0], deployments[cat_lookup][1]
                response = client.chat.completions.create(
                    model=name,
                    messages=[{"role": "system", "content": context},
                              {"role": "user",   "content": prompt}]
                )
                message = response.choices[0].message.content

                if cat == selected:
                    openai_models = {"gpt-3.5-turbo", "gpt-4"}
                    encodings     = {m: tiktoken.encoding_for_model(m) for m in openai_models}

                    try:
                        llm_llama_tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
                    except Exception:
                        llm_llama_tok = _fallback_tok

                    llm_name = arm_to_llm[selected]
                    text     = message if isinstance(message, str) else str(message)
                    if llm_name in encodings:
                        out_len = len(encodings[llm_name].encode(text))
                    else:
                        out_len = len(llm_llama_tok(text, truncation=True, padding=False)["input_ids"])

                if dataset == "telecom":
                    match = re.search(r"option \s*(\d+)", message)
                    label = re.search(r"option \s*(\d+)", labels[t])
                    if match is None:
                        match = re.search(r"Option \s*(\d+)", message)
                    if match:
                        num = int(match.group(1))
                        lab = int(label.group(1))
                        score = 1 if (lab == num) else 0
                        all_rewards_diag.append(score)
                        avg_array[name] = (avg_array[name] * t + score) / (t + 1)
                        result[name]    = score
                        if cat == selected or actual == selected:
                            reward = score

                elif dataset == "medical":
                    cnt = 0
                    if message is None:
                        message = ""
                    elif not isinstance(message, str):
                        if hasattr(message, "content"):
                            message = message.content
                        elif hasattr(message, "text"):
                            message = message.text
                        else:
                            message = str(message)
                    new_msg = message.lower()
                    for el in labels[t]:
                        if new_msg.find(el) != -1:
                            cnt += 1
                    score = cnt / len(labels[t])
                    all_rewards_diag.append(score)
                    avg_array[name] = (avg_array[name] * t + score) / (t + 1)
                    result[name]    = score
                    if cat == selected or actual == selected:
                        reward = score

                match = match  # exit while

    # Once we have `result[name]` for each arm, pick best_score and compute regret:
    if reward is None:
        raise ValueError("Reward was not calculated for the selected deployment.")
    if dataset == "telecom":
        best_score = 0
        best_dep   = None
        for dep, sc in result.items():
            if int(sc) >= best_score:
                best_score = int(sc)
                best_dep   = dep
        return int(best_score) - int(reward), int(reward), out_len, avg_array, all_rewards_sum, all_rewards_diag
    else:  # dataset == "medical"
        best_score = 0.0
        best_dep   = None
        for dep, sc in result.items():
            if float(sc) >= best_score:
                best_score = float(sc)
                best_dep   = dep
        return best_score - reward, reward, out_len, avg_array, all_rewards_sum, all_rewards_diag
