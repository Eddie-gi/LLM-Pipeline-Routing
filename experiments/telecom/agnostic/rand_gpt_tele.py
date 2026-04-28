import numpy as np
from src.prompts.prompt_maker import input_maker
from utils.helper import get_regret
import argparse

parser = argparse.ArgumentParser(description='NeuralUCB')
parser.add_argument('--size', default=250, type=int, help='number of rounds')
parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularzation')
parser.add_argument('--hidden', type=int, default=50, help='network hidden size')
parser.add_argument('--style', default='ucb', metavar='ts|ucb', help='TS or UCB')
parser.add_argument('--number_tasks', default=1)

args, unknown = parser.parse_known_args()
no_tasks = args.number_tasks
class DynamicVariables:
    pass

dyn_vars = DynamicVariables()
num_rounds = args.size


regrets = []
summ = 0
rew = 0
rewards_list = []
total_reward = 0
input_reports,labels = input_maker('rand',"telecom",total_reward)[0:args.size]
deployments_1 = {"base" : ("gpt-35-turbo","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."), "finetune_med" : ("Med","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."), "finetune_tele" : ("Tele","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."),"finetune_med_new" : ("Med_New","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."),"small": ("SLM","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 as there are 4 options for each question. Do not output an answer like 4 to indicate option 4."),"llama": ("llama","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 as there are 4 options for each question. Do not output an answer like 4 to indicate option 4."), "phi": ("phi","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 as there are 4 options for each question. Do not output an answer like 4 to indicate option 4.")}
emb_size = 384
deploy = [deployments_1]
cat = ''
rewards = 0
input_reports = list(input_reports)
from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
import random
dataset = "telecom"
plays_no = np.ones(len(deployments_1))
avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"SLM":0,"llama":0,"phi":0}
all_rewards_sum = []
all_rewards_diag = []
for t in range(num_rounds):
    context = []
    models = ["base","finetune_med","finetune_tele","finetune_med_new","small","llama","phi"]

    prompt_to_model = input_reports[t]
    task = 'diagnosis'
    values = []
    arm = random.randint(0,len(models)-1)
    plays_no[arm] += 1 
    arm_select = models[arm] 
    print(arm_select)
    
    if models[arm] == "finetune_med" or models[arm]=="finetune_tele" or models[arm]=="finetune_med_new":
        cat = "finetune"
        
    else:    
        cat = models[arm]
    dep = deploy[0]
    
    selected = arm_select
    fin_prompt = prompt_to_model
    deployment = dep[selected]
    reg,reward,avg_array,all_rewards_sum,all_rewards_diag = get_regret(deployments_1,fin_prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset) #fill arguments here
    rewards += int(reward)
    rewards_list.append(rewards)
    print("reward:")
    print(reward)
    print("reg:")
    print(reg)
    print(plays_no)
    print("Done")
    summ+= reg
    regrets.append(summ)
          
    if (t+1) % 5 == 0:
        print('{}: {:.3f}, {:.3f}'.format(t+1, summ, rewards))

import pickle
with open("regrets_random_tele_5_250.pkl", "wb") as file:
    pickle.dump(regrets, file)
with open("rewards_random_tele_5_250.pkl", "wb") as file:
    pickle.dump(rewards_list, file)
with open("arm_plays_random_tele_5_250.pkl", "wb") as file:
    pickle.dump(list(plays_no), file)
with open("avg_array_acc_llms_5_250_tele.pkl", "wb") as file:
    pickle.dump(avg_array, file)