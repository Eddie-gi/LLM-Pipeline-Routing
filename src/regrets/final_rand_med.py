from api.client import get_client
from openai import AssistantEventHandler, OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
import re
from transformers import AutoTokenizer

import tiktoken
'''
deployments structure

{
    assistants : [(names,context)],
    finetune : [(names,context)],
    base : [(names,context)]
}

Structured this way since different categories of llms were made in different resouce
groups when creating them in azure.
'''

#output is just the score for specific prompt
def final_eval(deployment, cat, prompt,task,all_rewards_sum,all_rewards_debate,all_rewards_diag,summary):
    name = deployment[0]
    context = deployment[1]
    number = -5000
    arm_to_llm = {
         "base"            : "gpt-3.5-turbo",
         "assistants"      : "gpt-3.5-turbo",
         "finetune_med"    : "gpt-4",
         "finetune_tele"   : "gpt-4",
         "finetune_med_new": "gpt-4",
         "llama"           : "llama-13b"
    }

    new_message = []
    if task == 'summary':
                new_message=[
                    {"role": "system", "content": " Your job is to rate the user's summary of a medical report based on the medical report provided and provide a numeric score which is strictly between 0 to 100 based on correctness and completeness. Use the following rubric: 90–100: Fully correct and comprehensive, 70–89: Mostly correct, may have small omissions, 50–69: Some correctness, but significant gaps or minor errors, 30–49: Limited correctness, missing major points or containing major errors, 10–29: Largely incorrect or irrelevant, 0–9: Completely nonsensical or irrelevant. After you decide the best score, explain your reasoning briefly, then output only: Score: <the numeric score>. Be thorough and do not inflate scores."},
                    {"role": "user", "content": f'The model context is: {context} output is: {prompt}.' }
                ]
    elif task == 'hallucination_pro':
        new_message = [
            {"role": "system",
                "content": "Your job is to rate the **pro argument** for factual correctness, on a 0–100 scale. "
                        "Use high scores sparingly if it’s tightly grounded in the summary."},
            {"role": "user",
                "content": f'The model context is: {context}. The PRO output is: {prompt}.'}
        ]
    elif task == 'hallucination_con':
        new_message = [
            {"role": "system",
                "content": "Your job is to rate the **con argument** for exposing hallucinations, "
                        "on a 0–100 scale. Use high scores sparingly if it pinpoints real errors."},
            {"role": "user",
                "content": f'The model context is: {context}. The CON output is: {prompt}.'}
        ]

    reg_model_name  = "bert-base-uncased"
    reg_tokenizer   = AutoTokenizer.from_pretrained(reg_model_name)

    while int(number) < 0 or int(number) > 100:
        if cat == 'assistants':
            new_client = get_client("base")
            response = new_client.chat.completions.create(
                model= "gpt-4o",
                messages=new_message
            )
            score = response.choices[0].message.content
            openai_models = {"gpt-3.5-turbo","gpt-4"}
            encodings = {m: tiktoken.encoding_for_model(m) for m in openai_models}
            try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
            except: llama_tok=reg_tokenizer
        
            # pick the right one for this selected arm
            llm_name = arm_to_llm[cat]
            if llm_name in encodings:
                out_len = len(encodings[llm_name].encode(summary))
            else:
                out_len = len(llama_tok(summary, truncation=True, padding=False)["input_ids"])
            number = ""
            match = re.search(r"Score:\s*(\d+)", score)
            if match:
                number = int(match.group(1))
                all_rewards_sum.append(number)
            if number == "":
                number = -5000

        elif cat == "small":
            new_client = get_client("base")
            response = new_client.chat.completions.create(
                model= "gpt-4o",
                messages=new_message
            )
            score = response.choices[0].message.content
            openai_models = {"gpt-3.5-turbo","gpt-4"}
            encodings = {m: tiktoken.encoding_for_model(m) for m in openai_models}
            try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
            except: llama_tok=reg_tokenizer
        
            # pick the right one for this selected arm
            llm_name = arm_to_llm[cat]
            if llm_name in encodings:
                out_len = len(encodings[llm_name].encode(summary))
            else:
                out_len = len(llama_tok(summary, truncation=True, padding=False)["input_ids"])
            number = ""
            match = re.search(r"Score:\s*(\d+)", score)
            if match:
                number = int(match.group(1))
                all_rewards_sum.append(number)
            if number == "":
                number = -5000
        
        elif cat == 'llama':
            new_client = get_client("base")
            response = new_client.chat.completions.create(
                model= "gpt-4o",
                messages=new_message
            )
            score = response.choices[0].message.content
            openai_models = {"gpt-3.5-turbo","gpt-4"}
            encodings = {m: tiktoken.encoding_for_model(m) for m in openai_models}
            try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
            except: llama_tok=reg_tokenizer
        
            # pick the right one for this selected arm
            llm_name = arm_to_llm[cat]
            if llm_name in encodings:
                out_len = len(encodings[llm_name].encode(summary))
            else:
                out_len = len(llama_tok(summary, truncation=True, padding=False)["input_ids"])
            number = ""
            match = re.search(r"Score:\s*(\d+)", score)
            if match:
                number = int(match.group(1))
                all_rewards_sum.append(number)
            if number == "":
                number = -5000
            

        else:
            new_client = get_client("base")
            response = new_client.chat.completions.create(
                model= "gpt-4o",
                messages=new_message
            )
            score = response.choices[0].message.content
            openai_models = {"gpt-3.5-turbo","gpt-4"}
            encodings = {m: tiktoken.encoding_for_model(m) for m in openai_models}
            try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
            except: llama_tok=reg_tokenizer
        
            # pick the right one for this selected arm
            llm_name = arm_to_llm[cat]
            if llm_name in encodings:
                out_len = len(encodings[llm_name].encode(summary))
            else:
                out_len = len(llama_tok(summary, truncation=True, padding=False)["input_ids"])
            number = ""
            match = re.search(r"Score:\s*(\d+)", score)
            if match:
                number = int(match.group(1))
                all_rewards_sum.append(number)
            if number == "":
                number = -5000

    print(f"summary reward: {int(number)}")
    return int(number),out_len,all_rewards_sum,all_rewards_debate,all_rewards_diag