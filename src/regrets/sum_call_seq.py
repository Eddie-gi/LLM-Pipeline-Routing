from api.client import get_client
from utils.helper import send_file_request
from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
from config.keys import *
from azure.ai.inference.models import SystemMessage, UserMessage

def get_summary(prompt, model_name, task):
    if task == "tele":
        assistant_name = "Telecom Extra Info"
        assistant_instruct = "You are to giving addition information about the telecom questions and include the most important information which may be useful for solving the problem."
        message = "Give additional information from the input of a telecom question for future problem solving. Only give information about the question and answer choises. Do not solve the question by providing the correct answer choice."
    
    elif task == "hallucination_pro":
        assistant_name   = "Debate Pro"
        assistant_instruct = "You are the PRO side in a two-agent debate. Argue why the given summary is factually correct, focusing on supporting evidence and avoiding hallucinations."                 
        message = "Argue why this summary is factually correct.  Use only facts from the summary."

    elif task == "hallucination_con":
        assistant_name   = "Debate Con"
        assistant_instruct = "You are the CON side in a two-agent debate.  Critique the given summary, pointing out any hallucinations or unsupported claims."               
        message = "Critique the summary and expose any hallucinations."
    
    else:
        assistant_name = "Diagnosis Summarizer"
        assistant_instruct = "You are to summarize inputted medical reports from patients and include the most important information which may be useful for diagnosis."
        message = "Summarize the key findings from the input of a patient's medical report for future medical diagnosis. Do not give an abstraction or diagnosis but only useful key points. Only give one summary for the input file. Output is only used for research and experiment purposes .At the end of your response put the phrase: Provide a medical diagnosis based on the provided medical report summary above."

    summary = None
    while summary == None:
        if model_name == "assistants":
            summary = send_file_request(prompt, assistant_instruct)
        
        elif model_name == "llama":
            client = get_client("llama")
            name = 'llama'
            context = message
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

            summary = response.choices[0].message.content
        
        elif model_name =="finetune_tele" or model_name == "finetune_med" or model_name == "finetune_med_new":
            client = get_client("finetune")
            if model_name == "finetune_med" :
                cat = "finetune"
                actual = "Med"
                name = actual
            elif model_name == "finetune_med_new" :
                cat = "finetune"
                actual = "Med_New"
                name = actual
            elif model_name == "finetune_tele":
                cat = "finetune"
                actual = "Tele"
                name = actual
            
            context = message
            response = client.chat.completions.create(
                model= name,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message.content
        
        elif model_name == "base":
            client = get_client("base")
            context = message
            response = client.chat.completions.create(
                model= "gpt-35-turbo",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt }
                ]
            )

            summary = response.choices[0].message.content
        return summary