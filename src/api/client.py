import os
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from config.keys import *

def get_client(cat):
    transport = RequestsTransport(connection_timeout=20, read_timeout=600)
    if cat == "assistants":
        client = AzureOpenAI(
            api_key=api_A,  
            api_version="2024-05-01-preview",
            azure_endpoint = endpoint_A
        )
    elif cat == "finetune":
        client = AzureOpenAI(
            azure_endpoint = endpoint_F, 
            api_key=api_F,  
            api_version="2024-02-01"
        )
    elif cat == "small":
        client = ChatCompletionsClient(
            endpoint=endpoint_S,
            credential=AzureKeyCredential(api_S),
            transport=transport
        )
    elif cat == "base":
        client = AzureOpenAI(
            azure_endpoint = endpoint_B, 
            api_key=api_B,  
            api_version="2024-08-01-preview"
        )
    elif cat == "llama":
        client = ChatCompletionsClient(
            endpoint=endpoint_L,
            credential=AzureKeyCredential(api_L),
            transport=transport
        )
    elif cat == "phi":
        client = ChatCompletionsClient(
            endpoint=endpoint_P,
            credential=AzureKeyCredential(api_P),
            transport=transport
        )
    elif cat == "phi_min":
        client = ChatCompletionsClient(
            endpoint=endpoint_PM,
            credential=AzureKeyCredential(api_PM),
        )
    elif cat == "deepseek":
        client = ChatCompletionsClient(
            endpoint=endpoint_D,
            credential=AzureKeyCredential(api_D),
        )
    else:
        raise Exception("Not recognized cat")
    return client