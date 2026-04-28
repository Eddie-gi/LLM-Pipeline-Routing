from api.client import get_client
from openai import AssistantEventHandler, OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
import re
import tiktoken
from transformers import AutoTokenizer

'''
deployments variable structure

{
    assistants : [(names,context)],
    finetune : [(names,context)],
    base : [(names,context)]
}

Structured this way since different categories of llms were made in different resouce
groups when creating them in azure.

output is a dict with model name as key and the evaluation score as value.
'''

def opt_eval(deployments, prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset):
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
    reward = None
    print(labels[t])
    reg_model_name  = "bert-base-uncased"
    reg_tokenizer   = AutoTokenizer.from_pretrained(reg_model_name)
    for cat in deployments:
        match = None
        if cat == "finetune_med" :
            cat = "finetune"
            actual = "Med"
        elif cat == "finetune_med_new" :
            cat = "finetune"
            actual = "Med_New"
        elif cat == "finetune_tele":
            cat = "finetune"
            actual = "Tele"
        client = get_client(cat)
        while match == None:
            if cat == 'assistants':

                for x in deployments[cat]:
                    name = x[0]
                    context = x[1]
                    assistant = client.beta.assistants.create(
                        name="Diagnosis Summarizer",
                        instructions=context,
                        tools=[{"type": "file_search"}],
                        model=name
                    )
    
                    vector_store = client.beta.vector_stores.create(name="Diagnosis Reports")
                    
                    f = open("data/input.txt", "w")
                    f.write(prompt)
                    f.close()
    
                    # Ready the files for upload to OpenAI
                    file_paths = ["data/input.txt"]
                    file_streams = [open(path, "rb") for path in file_paths]
                    
                    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
                    # and poll the status of the file batch for completion.
                    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store.id, files=file_streams
                    )
    
                    assistant = client.beta.assistants.update(
                        assistant_id=assistant.id,
                        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
                    )
    
                    # Upload the user provided file to OpenAI
                    message_file = client.files.create(
                    file=open("data/input.txt", "rb"), purpose="assistants"
                    )
                    
                    # Create a thread and attach the file to the message
                    thread = client.beta.threads.create(
                    messages=[
                        {
                        "role": "user",
                        "content": prompt,
                        # Attach the new file to the message.
                        "attachments": [
                            { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
                        ],
                        }
                    ]
                    )
                    
                    class EventHandler(AssistantEventHandler):
                        def on_message_done(self, message) -> None:
                            # print a citation to the file searched
                            message_content = message.content[0].text
                            annotations = message_content.annotations
                            citations = []
                            for index, annotation in enumerate(annotations):
                                message_content.value = message_content.value.replace(
                                    annotation.text, f"[{index}]"
                                )
                                if file_citation := getattr(annotation, "file_citation", None):
                                    cited_file = client.files.retrieve(file_citation.file_id)
                                    citations.append(f"[{index}] {cited_file.filename}")
    
                            f = open("data/outputs.txt", "w", encoding="utf-8")
                            f.write(message_content.value)
                            f.close()
    
                    with client.beta.threads.runs.stream(
                        thread_id=thread.id,
                        assistant_id=assistant.id,
                        instructions="Please address the user as Jane Doe. The user has a premium account.",
                        event_handler=EventHandler(),
                    ) as stream:
                        stream.until_done()
                    d = open("data/outputs.txt","r", encoding="utf-8")
                    message = d.read()
                    d.close()
                    if cat == selected:
                        openai_models = {"gpt-3.5-turbo","gpt-4"}
                        encodings = {m: tiktoken.encoding_for_model(m) for m in openai_models}
                        try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
                        except: llama_tok=reg_tokenizer
                    
                        # pick the right one for this selected arm
                        llm_name = arm_to_llm[selected]
                        if llm_name in encodings:
                            if not isinstance(message, str):
                                # if it has a `.content` or `.text` field, extract that first
                                if hasattr(message, "content"):
                                    message = message.content
                                elif hasattr(message, "text"):
                                    message = message.text
                                else:
                                    # fallback to Python’s built-in string conversion
                                    message = str(message)
                            
                            out_len = len(encodings[llm_name].encode(message))
                        else:
                            out_len = len(llama_tok(message, truncation=True, padding=False)["input_ids"])

                    if dataset == "telecom":
                        match = re.search(r"option \s*(\d+)", message)
                        label = re.search(r"option \s*(\d+)", labels[t])
                        if match == None:
                            match = re.search(r"Option \s*(\d+)", message)
                        if match:
                            num = int(match.group(1))
                            lab = int(label.group(1))
                            if lab == num:
                                number = 1
                            else:
                                number = 0
                            all_rewards_diag.append(number)
                            avg_array[name] = avg_array[name]*t + number
                            avg_array[name] = avg_array[name]/(t+1)
                            result[name] = number
                            if cat == selected or actual == selected:
                                reward = number
                    
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
                                cnt +=1
                            
                        number = cnt/len(labels[t])
                        all_rewards_diag.append(number)
                        avg_array[name] = avg_array[name]*t + number
                        avg_array[name] = avg_array[name]/(t+1)
                        result[name] = number
                        if cat == selected or actual == selected:
                            reward = number
                        match = True
                        
            elif cat == "small" or cat == "llama" or cat == "deepseek" or cat == "phi":

                name = deployments[cat][0]
                context = deployments[cat][1]
                message = None
                while message == None:
                    response = client.complete(
                        messages=[
                            SystemMessage(content=context),
                            UserMessage(content=prompt)
                        ],
                        max_tokens=2048,
                        temperature=0.8,
                        top_p=0.1,
                        model=name
                    )
        
                    message = response.choices[0].message.content
                if cat == selected:
                    print(message)
                    openai_models = {"gpt-3.5-turbo","gpt-4"}
                    encodings = {m: tiktoken.encoding_for_model(m) for m in openai_models}
                    try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
                    except: llama_tok=reg_tokenizer
                
                    # pick the right one for this selected arm
                    llm_name = arm_to_llm[selected]
                    if llm_name in encodings:
                        if not isinstance(message, str):
                            # if it has a `.content` or `.text` field, extract that first
                            if hasattr(message, "content"):
                                message = message.content
                            elif hasattr(message, "text"):
                                message = message.text
                            else:
                                # fallback to Python’s built-in string conversion
                                message = str(message)
                        
                        out_len = len(encodings[llm_name].encode(message))
                    else:
                        out_len = len(llama_tok(message, truncation=True, padding=False)["input_ids"])
                if dataset == "telecom":
                    if message == "{5}":
                        message = "option 5"
                    if message.isnumeric():
                        message = f"option {message}"
                    if message.find("{") != -1:
                        message = message.replace("{","")
                        message = message.replace("}","")
                        message = f"option {message}"
                    listt = ["1","2","3","4","5"]
                    for num in listt:
                        if message[0] == num:
                            message = f"option {num}"
                        elif message.find(f"{num}:")!=-1 or message.find(f"{num}: ")!=-1:
                            message = f"option {num}"
                        elif message.splitlines()[-1][0] == num:
                            message = f"option {num}"
                    match = re.search(r"option \s*(\d+)", message)
                    label = re.search(r"option \s*(\d+)", labels[t])
                    if match == None:
                        match = re.search(r"Option \s*(\d+)", message)
                    if match:
                        num = int(match.group(1))
                        lab = int(label.group(1))
                        if lab == num:
                            number = 1
                        else:
                            number = 0
                        all_rewards_diag.append(number)
                        avg_array[name] = avg_array[name]*t + number
                        avg_array[name] = avg_array[name]/(t+1)
                        result[name] = number
                        if cat == selected or actual == selected:
                            reward = number
                
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
                            cnt +=1
                    number = cnt/len(labels[t])
                    all_rewards_diag.append(number)
                    avg_array[name] = avg_array[name]*t + number
                    avg_array[name] = avg_array[name]/(t+1)
                    result[name] = number
                    if cat == selected or actual == selected:
                        reward = number
                    match = True
                    

            else:
                if actual == "Med":
                    cat = "finetune_med"
                elif actual =="Tele":
                    cat = "finetune_tele"
                elif actual == "Med_New":
                    cat = "finetune_med_new"
                name = deployments[cat][0]
                context = deployments[cat][1]
                message = None
                while message == None:
                    response = client.chat.completions.create(
                        model= name,
                        messages=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    message = response.choices[0].message.content
                if cat == selected:
                    print(message)
                    openai_models = {"gpt-3.5-turbo","gpt-4"}
                    encodings = {m: tiktoken.encoding_for_model(m) for m in openai_models}
                    try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
                    except: llama_tok=reg_tokenizer
                
                    # pick the right one for this selected arm
                    llm_name = arm_to_llm[selected]
                    if llm_name in encodings:
                        #out_len = len(encodings[llm_name].encode(message))
                        if not isinstance(message, str):
                            # if it has a `.content` or `.text` field, extract that first
                            if hasattr(message, "content"):
                                message = message.content
                            elif hasattr(message, "text"):
                                message = message.text
                            else:
                                # fallback to Python’s built-in string conversion
                                message = str(message)
                        
                        out_len = len(encodings[llm_name].encode(message))
                    else:
                        out_len = len(llama_tok(message, truncation=True, padding=False)["input_ids"])
                if dataset == "telecom":
                    match = re.search(r"option \s*(\d+)", message)
                    label = re.search(r"option \s*(\d+)", labels[t])
                    if match == None:
                        match = re.search(r"Option \s*(\d+)", message)
                    if match:
                        num = int(match.group(1))
                        lab = int(label.group(1))
                        if lab == num:
                            number = 1
                        else:
                            number = 0
                        all_rewards_diag.append(number)
                        avg_array[name] = avg_array[name]*t + number
                        avg_array[name] = avg_array[name]/(t+1)
                        result[name] = number
                        if cat == selected or actual == selected:
                            reward = number
                    
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
                            cnt +=1
                    number = cnt/len(labels[t])
                    all_rewards_diag.append(number)
                    avg_array[name] = avg_array[name]*t + number
                    avg_array[name] = avg_array[name]/(t+1)
                    result[name] = number
                    if cat == selected or actual == selected:
                        reward = number
                    match = True
                
            
            
    best_dep = ""
    best_score = 0
    print(result)
    print("avg:")
    print(avg_array)
    if reward is None:
        raise ValueError("Reward was not calculated for the selected deployment.")
    if dataset == "telecom":
        for dep in result:
            if int(result[dep]) >= best_score:
                best_score = int(result[dep])
                best_dep = dep
        return int(best_score)-int(reward),int(reward),out_len,avg_array,all_rewards_sum,all_rewards_diag
    elif dataset == "medical":
        best_dep = max(avg_array, key=lambda m: avg_array[m])
        best_inst = float(result.get(best_dep, 0.0))
        regret = best_inst - reward
        return regret, reward, out_len, avg_array, all_rewards_sum, all_rewards_diag
