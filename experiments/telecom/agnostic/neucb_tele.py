import numpy as np
from src.prompts.prompt_maker import input_maker
from utils.helper import get_regret  
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import argparse
from src.embedding.embed_tele import get_context

class Network(nn.Module):
    def __init__(self, dim = 100, hidden_size=100):
        super(Network, self).__init__()
        self.model = nn.Sequential(nn.Linear(dim, hidden_size),nn.ReLU(),nn.Linear(hidden_size, 1))


class NeuralUCBDiag:
    def __init__(self, style, dim, lamdba=1, nu=1, hidden=100):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device =='cuda':
            self.func = extend(Network(dim, hidden_size=hidden).model.cuda())
        else:
            self.func = Network(dim, hidden_size=hidden)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device =='cuda':
            self.U = lamdba * torch.ones((self.total_param,)).cuda()
            self.U_random = lamdba * torch.ones((self.total_param,)).cuda()
        else:
            self.U = lamdba * torch.ones((self.total_param,))
            self.U_random = lamdba * torch.ones((self.total_param,))
        self.nu = nu
        self.num_rounds = 100
        self.style = style
        
        if self.device =='cuda':
            self.loss_func = extend(nn.MSELoss().cuda())
        else:
            self.loss_func = extend(nn.MSELoss())
        self.len = 0

    def update_params(self,g_list):
        #print(g_list.size())
        for g in g_list:
            self.U += g * g
        return 0
    

    def selection(self,context,style):
        tensor = torch.from_numpy(np.array(context)).float().cuda()
        mu = self.func(tensor)
        self.func.zero_grad()
        mu.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
        sigma2 = self.lamdba * self.nu * g * g / self.U
        sigma = torch.sqrt(torch.sum(sigma2))
        if style == "ucb":
            sample_r = 0.2*mu.item() +2* sigma.item() 
        elif style == "ts":
            sample_r = torch.normal(1.5*mu.view(-1), 0.01*sigma.view(-1))

        self.U += g * g
        return sample_r
            
    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        optimizer = optim.SGD(self.func.parameters(), lr=1e-4, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                if self.device =='cuda':
                    delta = self.func(c.cuda()) - torch.tensor(float(r))
                else:
                    delta = self.func(c) - torch.tensor(float(r))
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 5:
                    return tot_loss / 5
            if batch_loss / length <= 1e-3:
                return batch_loss / length

parser = argparse.ArgumentParser(description='NeuralUCB')


parser.add_argument('--size', default=250, type=int, help='number of rounds')
parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularzation')
parser.add_argument('--hidden', type=int, default=50, help='network hidden size')
parser.add_argument('--style', default='ts', metavar='ts|ucb', help='TS or UCB')
parser.add_argument('--number_tasks', default=2)



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
input_reports,labels = input_maker('seq',"telecom",0)[0:args.size]
diagnosis_description_array = ["General use LLM which does not specialize in any task specifically.","LLM specializing on medical reports and trained to do medical diagnosis for research purposes.",
                               "LLM specializing for answering multiple choice telecommunications questions.", "LLM specializing on medical reports and trained to do medical diagnosis for research purposes.", "General use LLM which does not specialize in any task specifically.","General use LLM which does not specialize in any task specifically.", "General use LLM which does not specialize in any task specifically."]
deployments_1 = {"base" : ("gpt-35-turbo","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."), "finetune_med" : ("Med","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."), "finetune_tele" : ("Tele","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."),"finetune_med_new" : ("Med_New","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 if there are 4 options for each question. Do not output an answer like 4 to indicate option 4."),"small": ("SLM","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 as there are 4 options for each question. Do not output an answer like 4 to indicate option 4."),"llama": ("llama","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 as there are 4 options for each question. Do not output an answer like 4 to indicate option 4."), "phi": ("phi","You are to answer multiple choice questions related to telecommunications. Output your answer strictly as option {i} where i is between 1-4 as there are 4 options for each question. Do not output an answer like 4 to indicate option 4.")}


emb_size = 384
total_len = len(diagnosis_description_array)
l = NeuralUCBDiag(args.style,emb_size, args.lamdba, args.nu, args.hidden)

deploy = [deployments_1]
cat = ''
rewards = 0
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
input_reports = list(input_reports)
documents = diagnosis_description_array+input_reports
from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
from transformers import AutoTokenizer, AutoModel

diag_len = len(diagnosis_description_array)
all_rewards_sum = []
all_rewards_diag = []
plays_no = np.ones(total_len)
avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"SLM":0,"llama":0,"phi":0}
dataset = "telecom"
i = 0
for t in range(num_rounds):
    context = []
    models = ["base","finetune_med","finetune_tele","finetune_med_new","small","llama","phi"]

    prompt_to_model = input_reports[t]
    task = 'summary'
    values = []
    for j in range(len(diagnosis_description_array)):
        cont = get_context(documents,t,i,j,0,len(diagnosis_description_array),len(input_reports),inp_model,"telecom")
        context.append(cont)
        values.append(l.selection(cont,args.style))
    if args.style == "ts":
        values_np = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in values]
        arm = np.random.choice(np.where(np.array(values_np) == np.array(values_np).max())[0])
    elif args.style =="ucb":
        arm = np.argmax(values)

    plays_no[arm] += 1 
    arm_select = models[arm]
            

    print(values)
    print(arm_select)
    if models[arm] == "finetune_med" or models[arm]=="finetune_tele" or models[arm]=="finetune_med_new":
        cat = "finetune"
    else:    
        cat = models[arm]
    dep = deploy[0]

    selected = arm_select
    fin_prompt = prompt_to_model
    deployment = dep[selected]
    
    reg,reward,avg_array,all_rewards_sum,all_rewards_diag = get_regret(deployments_1,fin_prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset)
    rewards += int(reward)
    rewards_list.append(rewards)
    summ+= reg
    regrets.append(summ)
    print("reward:")
    print(reward)
    print("reg:")
    print(reg)
    print(plays_no)
    print("Done")
    
    if t ==0:
        if reward == 0:    
            all_rewards_diag.append(1)
        else:
            all_rewards_diag.append(0)
        new_rews = all_rewards_diag
        new_rews = (new_rews-np.mean(all_rewards_diag))/np.std(all_rewards_diag)
        index = 0
        loss = l.train(context[arm], new_rews[index])
    else:
        new_rews = all_rewards_diag
        new_rews = (new_rews-np.mean(all_rewards_diag))/np.std(all_rewards_diag)
        index = all_rewards_diag.index(reward)
        loss = l.train(context[arm], new_rews[index])

    if (t+1) % 5 == 0:
        print('{}: {:.3f}, {:.3f}, {:.3f}'.format(t+1, summ, rewards,loss))

import pickle
with open("regrets_neucb_tele_3_250.pkl", "wb") as file:
    pickle.dump(regrets, file)
with open("rewards_neucb_tele_3_250.pkl", "wb") as file:
    pickle.dump(rewards_list, file)
with open("arm_plays_neucb_tele_3_250.pkl", "wb") as file:
    pickle.dump(list(plays_no), file)
with open("avg_array_acc_llms_3_tele.pkl", "wb") as file:
    pickle.dump(avg_array, file)