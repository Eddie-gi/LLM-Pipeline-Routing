from azure.core.exceptions import HttpResponseError
import time

import numpy as np
from src.prompts.prompt_maker import input_maker
from utils.helper import opt_eval, get_summary
import argparse
import pickle
from src.regrets.final_rand_med import final_eval

def get_regret(deployments,prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset):
    return opt_eval(deployments, prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset)

parser = argparse.ArgumentParser(description='NeuralUCB')
parser.add_argument('--size', default=100, type=int, help='number of rounds')
parser.add_argument('--number_tasks', default=2)
parser.add_argument('--no_runs',   default=5,   type=int, help='how many independent runs')

def get_reward(deployment,cat,prompt,task,all_rewards_sum,all_rewards_diag,summary):
    
    #call Azure API to get final reward
    
    return final_eval(deployment, cat, prompt,task,all_rewards_sum,all_rewards_diag,summary)
    

with open('diagnoses_100.pkl', 'rb') as file: 

    # Call load method to deserialze 
    diagnoses = pickle.load(file)

with open('input_reports_100.pkl', 'rb') as file: 

    # Call load method to deserialze 
    input_reports = pickle.load(file)

new_labels = diagnoses

args, unknown = parser.parse_known_args()
no_tasks = args.number_tasks
no_runs = args.no_runs

num_rounds = args.size
all_regrets     = []   # list of length no_runs, each an array of length num_rounds
all_rewards     = []
all_costs       = []
all_costs_summarizer = []
all_plays       = []   # list of length no_runs, each an array of length num_arms
all_avg_arrays  = []   # list of dicts
all_avg_summary_arrays = []
all_runs_diag_stats = []

deployments_1 = {"base" : ("gpt-35-turbo","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med" : ("Med","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "finetune_tele" : ("Tele","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med_new": ("Med_New","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "llama": ("llama","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication")}

deployments_0 = {"base" : ("gpt-35-turbo","You are to summarize an inputted medical report, this summary will be used for research purposes only."), "assistants" : ("Assistant","You are to summarize an inputted medical report, this summary will be used for research purposes only."),"finetune_med" : ("Med","You are to summarize an inputted medical report, this summary will be used for research purposes only."),"finetune_tele" : ("Tele","You are to summarize an inputted medical report, this summary will be used for research purposes only."),"llama" : ("llama","You are to summarize an inputted medical report, this summary will be used for research purposes only."),}

emb_size = 384
deploy = [deployments_0,deployments_1]
cat = ''

arm_to_llm = {
    "base"            : "gpt-3.5-turbo",
    "assistants"      : "gpt-3.5-turbo",
    "finetune_med"    : "gpt-4",
    "finetune_tele"   : "gpt-4",
    "finetune_med_new": "gpt-4",
    "llama"           : "llama-13b"
}

input_cost_per_token = {
    "base"            : 0.0000005,
    "assistants"      : 0.0000005,
    "finetune_med"    : 0.00000025,
    "finetune_tele"   : 0.00000025,
    "finetune_med_new": 0.00000025,
    "llama"           : 0.00000071
}


cost_per_token = {
    "base"            : 0.0000015,   # GPT-3.5 Turbo
    "assistants"      : 0.0000015,   # GPT-3.5 Turbo
    "finetune_med"    : 0.00001,   # GPT-4
    "finetune_tele"   : 0.00001,   # GPT-4
    "finetune_med_new": 0.00001,   # GPT-4
    "llama"           : 0.00000071   # Llama-13b
}

        
dataset = "medical"
input_reports = input_maker("seq",dataset,input_reports)
import tiktoken

from transformers import AutoConfig, AutoTokenizer

from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
import random
dataset = "medical"

for run in range(no_runs):
    all_rewards_sum = []
    all_rewards_diag = []
    print(f"\n===== Starting run {run+1}/{no_runs} =====")
    regrets = []
    plays_no = np.ones(len(deployments_1)+len(deployments_0)) 
    summary_reward_sums = { arm: 0.0 for arm in deployments_0.keys() }
    summary_counts      = { arm:   0   for arm in deployments_0.keys() }
    summary_avg_array   = { arm: 0.0 for arm in deployments_0.keys() }
    diag_reward_by_summ   = { arm: []  for arm in deployments_0.keys() }
    diag_regret_by_summ   = { arm: []  for arm in deployments_0.keys() }

    summ = 0
    rew = 0
    rewards_list = []
    costs = 0
    costs_summarizer = 0
    costs_list = []
    costs_list_summarizer = []
    total_reward = 0
    avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"llama":0}
    rewards = 0
    openai_models = {"gpt-3.5-turbo","gpt-4"}
    encodings = { m: tiktoken.encoding_for_model(m) for m in openai_models }
    
    # for llama we use HuggingFace:
    llama_tok = AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
    
    # now build a lookup that is either a tiktoken encoder or HF tokenizer
    arm_encoders = {}
    for mk,llm in arm_to_llm.items():
        if llm in encodings:
            arm_encoders[mk] = encodings[llm]
        else:
            arm_encoders[mk] = llama_tok
            
    for t in range(num_rounds):
        for i in range(no_tasks):
            if i ==0:
                models = ["assistants","base","finetune_med","finetune_tele","llama"]
            
                prompt_to_model = input_reports[t]
                arm = random.randint(0,len(models)-1)
                plays_no[arm] += 1 
                arm_select = models[arm] 
                print(arm_select)
            
                if models[arm] == "finetune_med" or models[arm]=="finetune_tele" or models[arm]=="finetune_med_new":
                    cat = "finetune"
                    
                else:    
                    cat = models[arm]
                dep = deploy[i]
            
                in_lengths = []
                for mk in models:
                    enc = arm_encoders[mk]
                    # If this is a tiktoken Encoding, use .encode(...)
                    if hasattr(enc, "encode"):
                        # tiktoken Encoding
                        in_len = len(enc.encode(prompt_to_model))
                    else:
                        # HuggingFace tokenizer
                        in_len = len(enc(
                            prompt_to_model,
                            truncation=True,
                            padding=False
                        )["input_ids"])
                    in_lengths.append(in_len)
                print(f"Selected summarizer: {arm_select}")
                summarizer_choice = arm_select
                summary = get_summary(input_reports[t],summarizer_choice)
                prompt_to_model = input_reports[t] + " Below is the summary of this report:" + "\n\n" + summary
                deployment = dep[arm_select]
                fin_prompt=prompt_to_model
                task = 'summary'
                reward,out_len,all_rewards_sum,all_rewards_diag = get_reward(deployment,models[arm],fin_prompt,task,all_rewards_sum,all_rewards_diag,summary)
                costs_summarizer += input_cost_per_token[arm_select]*in_lengths[arm]+ cost_per_token[arm_select]* out_len
                costs_list_summarizer.append(costs_summarizer)  
                summary_reward_sums[arm_select] += reward
                summary_counts[arm_select]      += 1
                summary_avg_array[arm_select]    = (
                    summary_reward_sums[arm_select]
                    / summary_counts[arm_select]
                )
                print(f'Prompt input to sum evaluator: {prompt_to_model}')
            else:
                models = ["base","finetune_med","finetune_tele","finetune_med_new","llama"]
                fin_prompt = summary
                arm = random.randint(0,len(models)-1)
                plays_no[arm+len(deployments_0)] += 1 
                arm_select = models[arm] 
                print(f"Selected diagnoser: {arm_select}")
            
                if models[arm] == "finetune_med" or models[arm]=="finetune_tele" or models[arm]=="finetune_med_new":
                    cat = "finetune"
                    
                else:    
                    cat = models[arm]
                dep = deploy[i]
                print(f"Prompt to diagnoser: {fin_prompt}")
                in_lengths = []
                for mk in models:
                    enc = arm_encoders[mk]
                    # If this is a tiktoken Encoding, use .encode(...)
                    if hasattr(enc, "encode"):
                        # tiktoken Encoding
                        in_len = len(enc.encode(prompt_to_model))
                    else:
                        # HuggingFace tokenizer
                        in_len = len(enc(
                            prompt_to_model,
                            truncation=True,
                            padding=False
                        )["input_ids"])
                    in_lengths.append(in_len)
                deployment = dep[arm_select]
                selected= arm_select
                task = 'diagnosis'
                reg,reward,out_len,avg_array,all_rewards_sum,all_rewards_diag = get_regret(deployments_1,fin_prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,new_labels,dataset) #fill arguments here
                diag_reward_by_summ[summarizer_choice].append(reward)
                diag_regret_by_summ[summarizer_choice].append(reg)
                rewards += int(reward)
                rewards_list.append(rewards)
                costs += input_cost_per_token[arm_select]*in_lengths[arm]+ cost_per_token[arm_select]* out_len
                costs_list.append(costs)

                print(f"Reward (diagnoser): {reward}")
                print(f"Regret (diagnoser): {reg}")
                print(plays_no)
                summ+= reg
                regrets.append(summ)
        
        print(f"Round {t+1} summary averages: {summary_avg_array}")
        if (t+1) % 5 == 0:
            print('{}: {:.3f}, {:.3f}'.format(t+1, summ, rewards))
    all_regrets.append(regrets)
    all_rewards.append(rewards_list)
    all_costs.append(costs_list)
    all_costs_summarizer.append(costs_list_summarizer)
    all_plays.append(plays_no.copy())
    all_avg_arrays.append(avg_array.copy())
    all_avg_summary_arrays.append(summary_avg_array.copy())  
    all_diag_reward_means = {
        arm: np.mean(diag_reward_by_summ[arm]) 
            if diag_reward_by_summ[arm] else 0.0
        for arm in diag_reward_by_summ
    }
    all_diag_reward_stds  = {
        arm: np.std(diag_reward_by_summ[arm])  
            if diag_reward_by_summ[arm] else 0.0
        for arm in diag_reward_by_summ
    }
    # pack into one dict (or DataFrame row) and save
    all_runs_diag_stats.append({
        arm: (all_diag_reward_means[arm], all_diag_reward_stds[arm])
        for arm in all_diag_reward_means
    })  

import pandas as pd
regrets_arr = np.array(all_regrets)     # shape (no_runs, num_rounds)
rewards_arr = np.array(all_rewards)
costs_arr   = np.array(all_costs)
costs_summarizer_arr   = np.array(all_costs_summarizer)
plays_arr   = np.array(all_plays)       # shape (no_runs, num_arms)
avg_df      = pd.DataFrame(all_avg_arrays)  # columns=model names

mean_df = pd.DataFrame(
    { arm: [stats[arm][0] for stats in all_runs_diag_stats]
      for arm in diag_reward_by_summ }
)
std_df  = pd.DataFrame(
    { arm: [stats[arm][1] for stats in all_runs_diag_stats]
      for arm in diag_reward_by_summ }
)
overall_mean = mean_df.mean(axis=0).to_dict()
overall_std  = std_df.std(axis=0).to_dict()


regrets_mean = regrets_arr.mean(axis=0)
regrets_std  = regrets_arr.std(axis=0)
rewards_mean = rewards_arr.mean(axis=0)
rewards_std  = rewards_arr.std(axis=0)
costs_mean   = costs_arr.mean(axis=0)
costs_std    = costs_arr.std(axis=0)
costs_summarizer_mean   = costs_summarizer_arr.mean(axis=0)
costs_summarizer_std    = costs_summarizer_arr.std(axis=0)
plays_mean = plays_arr.mean(axis=0)
plays_std  = plays_arr.std(axis=0)
avg_mean = avg_df.mean(axis=0).to_dict()
avg_std  = avg_df.std(axis=0).to_dict()

summary_avg_df   = pd.DataFrame(all_avg_summary_arrays)
summary_mean     = summary_avg_df.mean(axis=0).to_dict()
summary_std      = summary_avg_df.std(axis=0).to_dict()

import pickle
pickle.dump(regrets_mean, open("regrets_mean_med_randgpt_4.pkl","wb"))
pickle.dump(regrets_std,  open("regrets_std_med_randgpt_4.pkl","wb"))
pickle.dump(rewards_mean, open("rewards_mean_med_randgpt_4.pkl","wb"))
pickle.dump(rewards_std,  open("rewards_std_med_randgpt_4.pkl","wb"))
pickle.dump(costs_mean,   open("costs_mean_med_randgpt_4.pkl","wb"))
pickle.dump(costs_std,    open("costs_std_med_randgpt_4.pkl","wb"))
pickle.dump(costs_summarizer_mean,   open("costs_summarizer_mean_med_randgpt_4.pkl","wb"))
pickle.dump(costs_summarizer_std,    open("costs_summarizer_std_med_randgpt_4.pkl","wb"))
pickle.dump(plays_mean,   open("plays_mean_med_randgpt_4.pkl","wb"))
pickle.dump(plays_std,    open("plays_std_med_randgpt_4.pkl","wb"))
pickle.dump(avg_mean,     open("avg_accuracy_mean_med_randgpt_4.pkl","wb"))
pickle.dump(avg_std,      open("avg_accuracy_std_med_randgpt_4.pkl","wb"))
pickle.dump(summary_mean,     open("sum_avg_accuracy_mean_med_randgpt_4.pkl","wb"))
pickle.dump(summary_std,      open("sum_avg_accuracy_std_med_randgpt_4.pkl","wb"))
pickle.dump(overall_mean,     open("summ_diag_med_eff_mean_rew_4_rand.pkl","wb"))
pickle.dump(overall_std,      open("summ_diag_med_eff_std_rew_4_rand.pkl","wb"))
print(f"Final mean regret: {regrets_mean[-1]}")
print(f"Final mean reward: {rewards_mean[-1]}")
print(f"Final mean cost: {costs_mean[-1]}")
print(f"Final mean summarizer cost: {costs_summarizer_mean[-1]}")
print(f"Final mean plays: {plays_mean}")
print(f"Final mean average array: {avg_mean}")
print(f"Final mean summary average array: {summary_mean}")
print(f"Final mean summary diag effect: {overall_mean}")

print("All runs complete. Summary pickles written.")