import os
import json
import openai
from openai import AzureOpenAI
from base_key import *
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def get_emb(documents,t,i,j,sum_descript_len,hall_descript_len,diag_descript_len,input_rep_len,model,dataset):
    if dataset == "medical":
        if i == 0:
            doc_id = j
            rep_id = t + sum_descript_len + hall_descript_len + diag_descript_len
        elif i == 1:
            doc_id = sum_descript_len + j
            rep_id = t + sum_descript_len + hall_descript_len + diag_descript_len
        else:
            doc_id = sum_descript_len + hall_descript_len + j
            rep_id = t + sum_descript_len + hall_descript_len + diag_descript_len
        emb_doc = model.encode(documents[doc_id],normalize_embeddings=False)
        emb_rep = model.encode(documents[rep_id],normalize_embeddings=False)
        return emb_doc,emb_rep
    elif dataset == "telecom":
        doc_id = j
        rep_id = t+diag_descript_len
        emb_doc = model.encode(documents[doc_id],normalize_embeddings=False)
        emb_rep = model.encode(documents[rep_id],normalize_embeddings=False)
        return emb_doc,emb_rep

        

def get_context(documents,t,i,j,sum_descript_len,hall_descript_len,diag_descript_len,input_rep_len,inp_model,dataset):
    descript_array_embed, task_array_embed = get_emb(documents,t,i,j,sum_descript_len,hall_descript_len,diag_descript_len,input_rep_len,inp_model,dataset)
    task_array = np.array(task_array_embed).astype(np.float64)  
    description_array = np.array(descript_array_embed).astype(np.float64)  

    context = task_array * description_array
    return context