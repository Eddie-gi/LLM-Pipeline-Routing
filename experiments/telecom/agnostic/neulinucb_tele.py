import numpy as np
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from src.prompts.prompt_maker import input_maker
import numpy as np
from src.regrets.optimal_rand_tele import opt_eval


def get_regret(deployments,prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset):
    return opt_eval(deployments, prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset)

def inv_sherman_morrison(u, A_inv):
	"""Inverse of a matrix with rank 1 update.
	"""
	Au = np.dot(A_inv, u)
	A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
	return A_inv
emb_size = 384
class Network(nn.Module):
	def __init__(self, dim, hidden_size=100):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(dim, hidden_size)
		self.activate = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, emb_size)
	def forward(self, x):
		return self.fc2(self.activate(self.fc1(x)))

class NeuralLinearUCB:
	def __init__(self, dim, lamdba=1, nu=1, hidden=100,n_arm = 7):
		self.n_arm = n_arm
		self.func = Network(dim, hidden_size=hidden).cuda()
		self.context_list = []
		self.arm_list = []
		self.reward = []
		self.lamdba = lamdba
		self.theta = np.random.uniform(-1, 1, (self.n_arm, dim))
		self.b = np.zeros((self.n_arm, dim))
		self.A_inv = np.array([np.eye(dim) for _ in range(self.n_arm)])

	def select(self, context):
		tensor = torch.from_numpy(context).float().cuda()
		features = self.func(tensor).cpu().detach().numpy()
		ucb = np.array([np.sqrt(np.dot(features[a,:], np.dot(self.A_inv[a], features[a,:].T))) for a in range(self.n_arm)])
		mu = np.array([np.dot(features[a,:], self.theta[a]) for a in range(self.n_arm)])
		xx = 0.08*mu + 0.08 * ucb
		print(xx)
		arm = np.random.choice(np.where(xx == xx.max())[0])
		return arm

	def train(self, context, arm_select, reward):
		self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
		self.arm_list.append(arm_select)
		self.reward.append(reward)
		optimizer = optim.SGD(self.func.parameters(), lr=1e-3, weight_decay=self.lamdba)
		length = len(self.reward)
		index = np.arange(length)
		np.random.shuffle(index)
		cnt = 0
		tot_loss = 0
		while True:
			batch_loss = 0
			for idx in index:
				c = self.context_list[idx]
				a = self.arm_list[idx]
				r = self.reward[idx]
				optimizer.zero_grad()
				features = self.func(c.cuda())
				mu = (features * torch.from_numpy(self.theta[a]).float().cuda()).sum(dim=1, keepdims=True)
				delta = mu - r
				loss = delta * delta
				loss.backward()
				optimizer.step()
				batch_loss += loss.item()
				tot_loss += loss.item()
				cnt += 1
				if cnt >= 5:
                    #print(c)
					return tot_loss / 5

	def update_model(self, context, arm_select, reward):
		tensor = torch.from_numpy(context).float().cuda()
		context = self.func(tensor).cpu().detach().numpy()
		self.theta = np.array([np.matmul(self.A_inv[a], self.b[a]) for a in range(self.n_arm)])
		self.b[arm_select] += context[arm_select] * reward
		self.A_inv[arm_select] = inv_sherman_morrison(context[arm_select,:],self.A_inv[arm_select])

parser = argparse.ArgumentParser(description='NeuralUCB')
parser.add_argument('--size', default=250, type=int, help='number of rounds')
parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularzation')
parser.add_argument('--hidden', type=int, default=50, help='network hidden size')
parser.add_argument('--style', default='ucb', metavar='ts|ucb', help='TS or UCB')
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

input_reports = list(input_reports)
documents = diagnosis_description_array+input_reports
from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
from embed_tele import get_context

diag_len = len(diagnosis_description_array)
all_rewards_sum = []
all_rewards_diag = []
total_len = len(diagnosis_description_array)
plays_no = np.ones(total_len)
avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"SLM":0,"llama":0,"phi":0}
l = NeuralLinearUCB(emb_size, 1, 1, 50,total_len)
i = 0 
rewards = 0
dataset = "telecom"
for t in range(num_rounds):
    context = []
    models = ["base","finetune_med","finetune_tele","finetune_med_new","small","llama","phi"]

    prompt_to_model = input_reports[t]
    task = 'summary'
    values = []
    
    for j in range(len(diagnosis_description_array)):
        cont = get_context(documents,t,i,j,0,len(diagnosis_description_array),len(input_reports),inp_model,"telecom")
        context.append(cont)
    arm_select = l.select(np.array(context))
    plays_no[arm_select] += 1 
    arm_selected = models[arm_select]
            

    print(arm_selected)
    if models[arm_select] == "finetune_med" or models[arm_select]=="finetune_tele" or models[arm_select]=="finetune_med_new":
        cat = "finetune"
    else:    
        cat = models[arm_select]
    dep = deployments_1

    selected = arm_selected
    fin_prompt = prompt_to_model
    deployment = dep[selected]
    reg,reward,avg_array,all_rewards_sum,all_rewards_diag = get_regret(deployments_1,fin_prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset)
    l.update_model(np.array(context), arm_select, reward)
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
        loss = l.train(context[arm_select], arm_select, new_rews[index])
    else:
        new_rews = all_rewards_diag
        new_rews = (new_rews-np.mean(all_rewards_diag))/np.std(all_rewards_diag)
        index = all_rewards_diag.index(reward)
        loss = l.train(context[arm_select], arm_select, new_rews[index])

    if (t+1) % 5 == 0:
        print('{}: {:.3f}, {:.3f}, {:.3f}'.format(t+1, summ, rewards,loss))

import pickle
with open("regrets_neulinucb_tele_5_250.pkl", "wb") as file:
    pickle.dump(regrets, file)
with open("rewards_neulinucb_tele_5_250.pkl", "wb") as file:
    pickle.dump(rewards_list, file)
with open("arm_plays_neulinucb_tele_5_250.pkl", "wb") as file:
    pickle.dump(list(plays_no), file)
with open("avg_array_acc_llms_5_tele_250.pkl", "wb") as file:
    pickle.dump(avg_array, file)