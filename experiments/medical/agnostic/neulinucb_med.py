import numpy as np
from utils.helper import get_optimal_super_arm_reward, get_reward, get_regret, parse_arguments, load_data
from src.prompts.prompt_maker import input_maker
from src.regrets.sum_call import get_summary
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import argparse
from src.embedding.embed_tele import get_context

# Parse command-line arguments
args = parse_arguments()
num_rounds = args.size
no_tasks = args.number_tasks

# Load data and process reports
diagnoses, input_reports = load_data()
input_reports = input_maker("rand", "medical", input_reports)
new_labels = diagnoses

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
	def __init__(self, dim, lamdba=1, nu=1, hidden=100, n_arm=2):
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
		xx = 0.2*mu + 0.5 * ucb
		print(xx)
		arm = np.random.choice(np.where(xx == xx.max())[0])
		return arm

	def train(self, context, arm_select, reward):
		self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
		self.arm_list.append(arm_select)
		self.reward.append(reward)
		optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
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

class DynamicVariables:
    pass

dyn_vars = DynamicVariables()

num_rounds = args.size


regrets = []
summ = 0
rew = 0
rewards_list = []
total_reward = 0
dataset = "medical"
input_reports = input_maker("",dataset,input_reports)
summary_description_array = ["Summarize the main points of the medical report of a patient, this summary will be used for research purposes only.",
                             "You will take the role of an diagnosis analyst. Use your knowledge base to summarize inputted medical reports."]
diagnosis_description_array = ["General use LLM which does not specialize in any task specifically.","LLM specializing on medical reports and trained to do medical diagnosis for research purposes.",
                               "LLM specializing for answering multiple choice telecommunications questions.", "LLM specializing on medical reports and trained to do medical diagnosis for research purposes.", "General use small LLM which does not specialize in any task specifically.","General use LLM which does not specialize in any task specifically.", "General use LLM which does not specialize in any task specifically."]
deployments_0 = {"base" : ("gpt-35-turbo","You are to summarize an inputted medical report, this summary will be used for research purposes only."), "assistants" : ("Assistant","You are to summarize an inputted medical report, this summary will be used for research purposes only.")}
deployments_1 = {"base" : ("gpt-35-turbo","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med" : ("Med","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "finetune_tele" : ("Tele","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med_new": ("Med_New","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "small": ("SLM","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "llama": ("llama","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "phi": ("phi","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication")}
emb_size = 384
total_len = len(summary_description_array)+len(diagnosis_description_array)
for i in range(no_tasks):
    if i == 0:
        setattr(dyn_vars, f'l_{i}', NeuralLinearUCB(emb_size, args.lamdba, args.nu, args.hidden,len(summary_description_array)))
    else:
        setattr(dyn_vars, f'l_{i}', NeuralLinearUCB(emb_size, args.lamdba, args.nu, args.hidden,len(diagnosis_description_array)))

deploy = [deployments_0,deployments_1]
cat = ''
rewards = 0
input_reports = list(input_reports)
sum_len = len(summary_description_array)
documents = summary_description_array+diagnosis_description_array+input_reports
from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

diag_len = len(diagnosis_description_array)
all_rewards_sum = []
all_rewards_diag = []
plays_no = np.ones(total_len)
avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"SLM":0,"llama":0,"phi":0}
dataset = "medical"
import random
for t in range(num_rounds):
    for i in range(no_tasks):
        context = []
        if i==0:
            models = ["assistants","base"]
        
            prompt_to_model = input_reports[t]
            task = 'summary'
            values = []
            l = getattr(dyn_vars, f'l_{i}')
            for j in range(len(summary_description_array)):
                
                cont = get_context(documents,t,i,j,len(summary_description_array),len(diagnosis_description_array),len(input_reports),inp_model,dataset)
                context.append(cont)
                
            arm = random.randint(0,len(models)-1)
            
            plays_no[arm] += 1 
            arm_select = models[arm]
        
        else:
            models = ["base","finetune_med","finetune_tele","finetune_med_new","small","llama","phi"]
            values=[]
            task = 'diagnosis'
            prompt_to_model = get_summary(input_reports[t],cat)
            prompt_to_model = prompt_to_model.replace('\n','')
            print(prompt_to_model)
            documents = documents+[prompt_to_model]
            l = getattr(dyn_vars, f'l_{i}')
            for j in range(len(diagnosis_description_array)):
                
                cont = get_context(documents,t,i,j,len(summary_description_array),len(diagnosis_description_array),len(input_reports),inp_model,dataset)
                context.append(cont)
                
            arm = l.select(np.array(context))
            plays_no[arm+sum_len] += 1 
            arm_select = models[arm]
    
        print(arm_select)
        
        if models[arm] == "finetune_med" or models[arm]=="finetune_tele" or models[arm]=="finetune_med_new":
            cat = "finetune"
        else:    
            cat = models[arm]
        dep = deploy[i]
    
        selected = arm_select
        fin_prompt = prompt_to_model
        deployment = dep[selected]
        
        if i==no_tasks-1:
            reg,reward,avg_array,all_rewards_sum,all_rewards_diag = get_regret(deployments_1,fin_prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,new_labels, dataset) #fill arguments here
            
            rewards += int(reward)
            rewards_list.append(rewards)
            print(reward)
            print(reg)
            print(plays_no)
            print("Done")
            summ+= reg
            regrets.append(summ)
            l = getattr(dyn_vars, f'l_{i}')
            l.update_model(np.array(context), arm, reward)
            new_rews = all_rewards_diag
            if t!= 0:
                new_rews = (new_rews-np.mean(all_rewards_diag))/np.std(all_rewards_diag)
            index = all_rewards_diag.index(reward)
            loss = l.train(context[arm], arm, new_rews[index])

        else:
            reward,all_rewards_sum,all_rewards_diag = get_reward(deployment,cat,fin_prompt,task,all_rewards_sum,all_rewards_diag)
            l = getattr(dyn_vars, f'l_{i}')
            l.update_model(np.array(context), arm, reward)

            if t ==0:
                all_rewards_sum.append(reward+1)
                new_rews = all_rewards_sum
                index = all_rewards_sum.index(reward)
                loss = l.train(context[arm], arm, new_rews[index]/100)
            else:
                new_rews = all_rewards_sum
                new_rews = (new_rews-np.mean(all_rewards_sum))/np.std(all_rewards_sum)
                index = all_rewards_sum.index(reward)
                loss = l.train(context[arm],arm, new_rews[index])

    if (t+1) % 5 == 0:
        print('{}: {:.3f}, {:.3f}, {:.3f}'.format(t+1, summ, rewards,loss))

import pickle
with open("regrets_neulinucb_med_100_4.pkl", "wb") as file:
    pickle.dump(regrets, file)
with open("rewards_neulinucb_med_100_4.pkl", "wb") as file:
    pickle.dump(rewards_list, file)
with open("arm_plays_neulinucb_med_100_4.pkl", "wb") as file:
    pickle.dump(list(plays_no), file)
with open("avg_array_acc_llms_med_100_4.pkl", "wb") as file:
    pickle.dump(avg_array, file)