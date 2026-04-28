import numpy as np
from src.regrets.final_rand_med import final_eval
from src.prompts.prompt_maker_seq import input_maker
from utils.helper import opt_eval, get_summary
import pickle
from transformers import AutoConfig, AutoTokenizer
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import argparse
from src.embedding.embed_med import get_context
from src.token_prediction.tok_length_predict import BertRegressionModel  

with open('diagnoses_100.pkl', 'rb') as file: 

    # Call load method to deserialze 
    diagnoses = pickle.load(file)

with open('input_reports_100.pkl', 'rb') as file: 

    # Call load method to deserialze 
    input_reports = pickle.load(file)

new_labels = diagnoses

def get_optimal_super_arm_reward(deployments,prompt,task):
    
    #call Azure API to get optimal reward by trying out all combinations

    return opt_eval(deployments, prompt,task)

def get_reward(deployment,cat,prompt,task,all_rewards_sum,all_rewards_debate,all_rewards_diag,summary):    
    return final_eval(deployment, cat, prompt,task,all_rewards_sum,all_rewards_debate,all_rewards_diag,summary)
    

def get_regret(deployments,prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset):
    return opt_eval(deployments, prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset)

reg_model_name  = "bert-base-uncased"
reg_config      = AutoConfig.from_pretrained(reg_model_name)
reg_tokenizer   = AutoTokenizer.from_pretrained(reg_model_name)
import json
with open("model_names.json") as f:
    orig_model_names = json.load(f)   # e.g. ["RWKV-4-Raven-14B","alpaca-13b",…,"gpt-4",…,"gpt-3.5-turbo",…,"llama-2-7b-chat",…]
num_models = len(orig_model_names)    
device = "cuda" if torch.cuda.is_available() else "cpu"
token_length_model = BertRegressionModel(reg_config, reg_model_name,hidden_dim=128,num_models=num_models).to(device)
token_length_model.load_state_dict(
    torch.load("best_length_model.pth", map_location=device)
)
token_length_model.eval()
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

class Network(nn.Module):
    def __init__(self, dim = 100, hidden_size=100):
        super(Network, self).__init__()

        self.model = nn.Sequential(nn.Linear(dim, hidden_size),nn.ReLU(),nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.model(x)


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
        for g in g_list:
            self.U += g * g
        return 0
    
    
    def selection(self,context,style):
        tensor = torch.from_numpy(np.array(context)).float()
        mu = self.func(tensor)
        self.func.zero_grad()
        mu.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
        sigma2 = self.lamdba * self.nu * g * g / self.U
        sigma = torch.sqrt(torch.sum(sigma2))
        if style == 'ucb':
            sample_r = mu.item() +0.5* sigma.item() 
        else:
            std = 0.7 * sigma
            std_val = max(std.item(), 0.0)
            noise = torch.randn(1).item()    # a single scalar N(0,1)
            sample_r = mu.item() + noise * std_val
        self.U += g * g
        return sample_r
    
    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
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

 
parser.add_argument('--size', default=100, type=int, help='number of rounds')
parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularzation')
parser.add_argument('--hidden', type=int, default=50, help='network hidden size')
parser.add_argument('--style', default='ts', metavar='ts|ucb', help='TS or UCB')
parser.add_argument('--number_tasks', default=3)
parser.add_argument('--no_runs',   default=5,   type=int, help='how many independent runs')
parser.add_argument('--alpha',   default=100,   type=int, help='cost accuracy tradeoff weight')



args, unknown = parser.parse_known_args()
no_tasks = args.number_tasks
no_runs = args.no_runs
alpha = args.alpha
all_regrets     = []   # list of length no_runs, each an array of length num_rounds
all_rewards     = []
all_costs       = []
all_costs_summarizer = []
all_costs_debate = []
all_plays       = []   # list of length no_runs, each an array of length num_arms
all_avg_arrays  = []   # list of dicts
all_avg_summary_arrays = []

class DynamicVariables:
    pass

num_rounds = args.size

summary_description_array = ["You will take the role of an diagnosis analyst. Use your knowledge base to summarize inputted medical reports.","General use LLM which does not specialize in any task specifically.","LLM specializing on medical reports and trained to do medical diagnosis for research purposes.",
                               "LLM specializing for answering multiple choice telecommunications questions.", "General use LLM which does not specialize in any task specifically."]

hallucination_description_array = [
    "Argue why this summary is factually correct. Output is used for research purposes only.",
    "Critique the summary and expose any possible hallucinations. Output is used for research purposes only."
]

diagnosis_description_array = ["General use LLM which does not specialize in any task specifically.","LLM specializing on medical reports and trained to do medical diagnosis for research purposes.",
                               "LLM specializing for answering multiple choice telecommunications questions.", "LLM specializing on medical reports and trained to do medical diagnosis for research purposes.", "General use LLM which does not specialize in any task specifically."]

deployments_1 = {"base" : ("gpt-35-turbo","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med" : ("Med","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "finetune_tele" : ("Tele","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med_new": ("Med_New","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "llama": ("llama","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication")}


deployments_0 = {"base" : ("gpt-35-turbo","You are to summarize an inputted medical report, this summary will be used for research purposes only."), "assistants" : ("Assistant","You are to summarize an inputted medical report, this summary will be used for research purposes only."),"finetune_med" : ("Med","You are to summarize an inputted medical report, this summary will be used for research purposes only."),"finetune_tele" : ("Tele","You are to summarize an inputted medical report, this summary will be used for research purposes only."),"llama" : ("llama","You are to summarize an inputted medical report, this summary will be used for research purposes only."),}

deployments_hall = deployments_0.copy()

emb_size = 384
total_len = len(deployments_0)+len(deployments_0)+len(deployments_1)
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


deploy = [deployments_0,deployments_hall,deployments_1]
cat = ''

input_reports = list(input_reports)
documents = summary_description_array+summary_description_array+diagnosis_description_array+input_reports
from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


import tiktoken
cost_error_dict = { arm: [] for arm in arm_to_llm.keys() }

sum_len = len(summary_description_array)

for run in range(no_runs):
    print(f"\n===== Starting run {run+1}/{no_runs} =====")
    regrets = []
    costs_list = []
    costs = 0
    costs_summarizer = 0
    costs_list_summarizer = []
    costs_halluc = 0
    costs_list_halluc = []
    dyn_vars = DynamicVariables()
    all_rewards_sum = []
    all_rewards_debate = []
    all_rewards_diag = []
    plays_no = np.ones(total_len)
    summary_reward_sums = { arm: 0.0 for arm in deployments_0.keys() }
    summary_counts      = { arm:   0   for arm in deployments_0.keys() }
    summary_avg_array   = { arm: 0.0 for arm in deployments_0.keys() }
    summ = 0
    rew = 0
    rewards_list = []
    rewards = 0
    total_reward = 0
    avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"llama":0}
    for i in range(no_tasks):
        setattr(dyn_vars, f'l_{i}', NeuralUCBDiag(args.style,emb_size, args.lamdba, args.nu, args.hidden))
        
    openai_models = {"gpt-3.5-turbo","gpt-4"}
    encodings = { m: tiktoken.encoding_for_model(m) for m in openai_models }
    
    # for llama we use HuggingFace:
    try: llama_tok=AutoTokenizer.from_pretrained('openlm-research/open_llama_13b')
    except: llama_tok=reg_tokenizer
    
    # now build a lookup that is either a tiktoken encoder or HF tokenizer
    arm_encoders = {}
    for mk,llm in arm_to_llm.items():
        if llm in encodings:
            arm_encoders[mk] = encodings[llm]
        else:
            arm_encoders[mk] = llama_tok
    for t in range(num_rounds):
        print(f"Round {t+1}")
        for i in range(no_tasks):
            if i==0:
                context = []
                models = ["base","assistants","finetune_med","finetune_tele","llama"]
                prompt_to_model = input_reports[t]
                task = 'summary'
                values = []
                l = getattr(dyn_vars, f'l_{i}')
                for j in range(len(summary_description_array)):
                    cont = get_context(documents,t,i,j,len(summary_description_array),len(summary_description_array),len(diagnosis_description_array),len(input_reports),inp_model,dataset)
                    context.append(cont)
                    values.append(l.selection(cont,args.style))
                toks = reg_tokenizer(
                    prompt_to_model,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt"
                ).to(device) 
                pred_lengths = []
                for mk in models:                   # models = ["base",…,"llama"]
                   llm_name = arm_to_llm[mk]       # e.g. "gpt-4"
                   idx      = orig_model_names.index(llm_name)
                   onehot   = torch.zeros(len(orig_model_names), device=device)
                   onehot[idx] = 1.0
                   onehot   = onehot.unsqueeze(0)  # [1×25]
            
                   with torch.no_grad():
                       pred = token_length_model(
                           toks["input_ids"],
                           toks["attention_mask"],
                           onehot
                       )
                   pred_lengths.append(pred.item())
                
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
                pred_cost = [
                    input_cost_per_token[mk] * in_lengths[j]
                    + cost_per_token[mk]  * pred_lengths[j]
                    for j, mk in enumerate(models)
                ]
                print(f"Pred cost summarizer: {pred_cost}")
                cost = []
                values_f = [float(v) for v in values]
                print(f"Accuracy UCB (summarizer): {values_f}")
                for arm_idx, mk in enumerate(models):
                    values[arm_idx] = (
                        values[arm_idx]- alpha*cost_per_token[mk]* pred_lengths[arm_idx]- alpha*input_cost_per_token[mk] * in_lengths[arm_idx]
                    )
                    cost.append(alpha*cost_per_token[mk]* pred_lengths[arm_idx] + alpha*input_cost_per_token[mk] * in_lengths[arm_idx])
                print(f"Budget cost (summarizer): {cost}")
                
                
                if args.style == "ts":
                    values_np = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in values]
                    arm = np.random.choice(np.where(np.array(values_np) == np.array(values_np).max())[0])
                elif args.style =="ucb":
                    arm = np.argmax(values)
            
                plays_no[arm] += 1 
                arm_select = models[arm]
                values_f_1 = [float(v) for v in values]

                print(f"Final values (summarizer): {values_f_1}")
                print(f"Selected summarizer: {arm_select}")
                summarizer_choice = arm_select
                summary = get_summary(input_reports[t],summarizer_choice, "Summary")
                prompt_to_model = input_reports[t] + " Below is the summary of this report:" + "\n\n" + summary

            elif i == 1:
                context = []
                models = ["base","assistants","finetune_med","finetune_tele","llama"]
                prompt_to_model = summary
                prompt_to_model = prompt_to_model.replace('\n','')
                task = 'hallucination'
                values = []
                documents = documents+[prompt_to_model]
                l = getattr(dyn_vars, f'l_{i}')
                for j in range(len(summary_description_array)):
                    cont = get_context(documents,t,i,j,len(summary_description_array),len(summary_description_array),len(diagnosis_description_array),len(input_reports),inp_model,dataset)
                    context.append(cont)
                    values.append(l.selection(cont,args.style))
                toks = reg_tokenizer(
                    prompt_to_model,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt"
                ).to(device) 
                pred_lengths = []
                for mk in models:                   # models = ["base",…,"llama"]
                   llm_name = arm_to_llm[mk]       # e.g. "gpt-4"
                   idx      = orig_model_names.index(llm_name)
                   onehot   = torch.zeros(len(orig_model_names), device=device)
                   onehot[idx] = 1.0
                   onehot   = onehot.unsqueeze(0)  # [1×25]
            
                   with torch.no_grad():
                       pred = token_length_model(
                           toks["input_ids"],
                           toks["attention_mask"],
                           onehot
                       )
                   pred_lengths.append(pred.item())
                
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
                pred_cost = [
                    input_cost_per_token[mk] * in_lengths[j]
                    + cost_per_token[mk]  * pred_lengths[j]
                    for j, mk in enumerate(models)
                ]
                print(f"Pred cost debater: {pred_cost}")
                cost = []
                values_f = [float(v) for v in values]
                print(f"Accuracy UCB (debater): {values_f}")
                for arm_idx, mk in enumerate(models):
                    values[arm_idx] = (
                        values[arm_idx]- alpha*cost_per_token[mk]* pred_lengths[arm_idx]- alpha*input_cost_per_token[mk] * in_lengths[arm_idx]
                    )
                    cost.append(alpha*cost_per_token[mk]* pred_lengths[arm_idx] + alpha*input_cost_per_token[mk] * in_lengths[arm_idx])
                print(f"Budget cost (debater): {cost}")
                
                
                if args.style == "ts":
                    values_np = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in values]
                    arm = np.random.choice(np.where(np.array(values_np) == np.array(values_np).max())[0])
                elif args.style =="ucb":
                    arm = np.argmax(values)
            
                plays_no[arm+sum_len] += 1 
                arm_select = models[arm]
                values_f_1 = [float(v) for v in values]

                print(f"Final values (debater): {values_f_1}")
                print(f"Selected debater: {arm_select}")
                dep = deploy[i]               
                selected = arm_select            
                deployment = dep[selected]

                # 1) PRO argument
                prompt_to_model_pro = prompt_to_model + hallucination_description_array[0]
                pro_out = get_summary(prompt_to_model_pro, arm_select, "hallucination_pro")
                r_pro, out_len_pro, *_ = get_reward(
                    deployment, arm_select, prompt_to_model_pro,
                    'hallucination_pro',
                    all_rewards_sum, all_rewards_debate, all_rewards_diag, pro_out
                )
                # 2) CON argument
                prompt_to_model_con = prompt_to_model + hallucination_description_array[1]
                con_out = get_summary(prompt_to_model_con, arm_select, "hallucination_con")
                r_con, out_len_con, *_ = get_reward(
                    deployment, arm_select, prompt_to_model_con,
                    'hallucination_con',
                    all_rewards_sum, all_rewards_debate, all_rewards_diag, con_out
                )

                # combine reward and track cost
                debate_reward = r_pro - r_con
                costs_halluc += cost_per_token[arm_select] * (out_len_pro + out_len_con) + input_cost_per_token[arm_select] * (in_lengths[arm] * 2)
                costs_list_halluc.append(costs_halluc)

                # forward winning text into diagnosis
                if (r_pro > r_con):
                    debate = pro_out
                    prompt_to_model =  prompt_to_model_pro 
                else:
                    debate = con_out
                    prompt_to_model =  prompt_to_model_con

                
            else:
                context = []
                values=[]
                models = ["base","finetune_med","finetune_tele","finetune_med_new","llama"]
                prompt_to_model = prompt_to_model.replace('\n','')
                task = 'diagnosis'
                documents = documents+[prompt_to_model]
                l = getattr(dyn_vars, f'l_{i}')
                for j in range(len(diagnosis_description_array)):
                    cont = get_context(documents,t,i,j,len(summary_description_array),len(summary_description_array),len(diagnosis_description_array),len(input_reports),inp_model,dataset)
                    context.append(cont)
                    values.append(l.selection(cont,args.style)) 
                    
                toks = reg_tokenizer(
                    prompt_to_model,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt"
                ).to(device) 
                pred_lengths = []
                for mk in models:                   # models = ["base",…,"llama"]
                   llm_name = arm_to_llm[mk]       # e.g. "gpt-4"
                   idx      = orig_model_names.index(llm_name)
                   onehot   = torch.zeros(len(orig_model_names), device=device)
                   onehot[idx] = 1.0
                   onehot   = onehot.unsqueeze(0)  # [1×25]
            
                   with torch.no_grad():
                       pred = token_length_model(
                           toks["input_ids"],
                           toks["attention_mask"],
                           onehot
                       )
                   pred_lengths.append(pred.item())
                pred_cost = [
                    input_cost_per_token[mk] * in_lengths[j]
                    + cost_per_token[mk]  * pred_lengths[j]
                    for j, mk in enumerate(models)
                ]
                print(f"Pred cost diagnoser: {pred_cost}")
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
                cost = []
                values_f = [float(v) for v in values]
                print(f"Accuracy UCB (diagnoser): {values_f}")
                for arm_idx, mk in enumerate(models):
                    values[arm_idx] = (
                        values[arm_idx]- alpha*cost_per_token[mk]* pred_lengths[arm_idx]- alpha*input_cost_per_token[mk] * in_lengths[arm_idx]
                    )
                    cost.append(alpha*cost_per_token[mk]* pred_lengths[arm_idx] + alpha*input_cost_per_token[mk] * in_lengths[arm_idx])
                print(f"Budget cost (diagnoser): {cost}")
                
                
                if args.style == "ts":
                    values_np = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in values]
                    arm = np.random.choice(np.where(np.array(values_np) == np.array(values_np).max())[0])
                elif args.style =="ucb":
                    arm = np.argmax(values)
            
                plays_no[arm+sum_len+sum_len] += 1 
                arm_select = models[arm]
                values_f_1 = [float(v) for v in values]

                print(f"Final values (diagnoser): {values_f_1}")
                print(f"Selected diagnoser: {arm_select}")    
                    
                    
            if models[arm] == "finetune_med" or models[arm]=="finetune_tele" or models[arm] == "finetune_med_new":
                cat = "finetune"
            else:    
                cat = models[arm]
            dep = deploy[i]
            selected = arm_select
            fin_prompt = prompt_to_model
            deployment = dep[selected]
    
            if i==no_tasks-1:
                fin_prompt = summary
                reg,reward,out_len,avg_array,all_rewards_sum,all_rewards_diag = get_regret(deployments_1,fin_prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,new_labels, dataset) #fill arguments here
                costs += input_cost_per_token[arm_select]*in_lengths[arm]+ cost_per_token[arm_select]* out_len
                costs_list.append(costs)
                actual_cost = (
                    input_cost_per_token[arm_select] * in_lengths[arm]
                    + cost_per_token[arm_select] * out_len
                )
                print(f"Actual cost diagnoser: {actual_cost}")
                error = (pred_cost[arm] - actual_cost)
                cost_error_dict[arm_select].append(error)
                rewards += int(reward)
                rewards_list.append(rewards)
                print(f'Reward: {reward}')
                print(f'Regret: {reg}')
                print(plays_no)
                print("Done")
                summ+= reg
                regrets.append(summ)
                l = getattr(dyn_vars, f'l_{i}')
                new_rews = all_rewards_diag
                if t!= 0:
                    std = np.std(all_rewards_diag)
                    if std < 1e-8:
                        std = 1.0  # fallback to avoid NaN
                    new_rews = (new_rews - np.mean(all_rewards_diag)) / std
                index = all_rewards_diag.index(reward)
                loss = l.train(context[arm], new_rews[index])

            elif i == 1:
                l = getattr(dyn_vars, f'l_{i}')
                debate_reward = 0.5 * debate_reward + 50
                all_rewards_debate.append(debate_reward)
                if t ==0:
                    print(all_rewards_debate)
                    new_rews = all_rewards_debate
                    index = all_rewards_debate.index(debate_reward)
                    loss = l.train(context[arm], new_rews[index]/1000)
                else:
                    new_rews = all_rewards_debate
                    new_rews = (new_rews-np.mean(all_rewards_debate))/np.std(all_rewards_debate)
                    index = all_rewards_debate.index(debate_reward)
                    loss = l.train(context[arm], new_rews[index])
            
            else:
                reward,out_len,all_rewards_sum,all_rewards_debate,all_rewards_diag = get_reward(deployment,models[arm],fin_prompt,task,all_rewards_sum,all_rewards_debate,all_rewards_diag,summary)
                l = getattr(dyn_vars, f'l_{i}')
                costs_summarizer += input_cost_per_token[arm_select]*in_lengths[arm]+ cost_per_token[arm_select]* out_len
                costs_list_summarizer.append(costs_summarizer)
                actual_cost = (
                    input_cost_per_token[arm_select] * in_lengths[arm]
                    + cost_per_token[arm_select] * out_len
                )
                print(f"Actual cost summarizer: {actual_cost}")
                error = (pred_cost[arm] - actual_cost)
                cost_error_dict[arm_select].append(error)
                summary_reward_sums[arm_select] += reward
                summary_counts[arm_select]      += 1
                summary_avg_array[arm_select]    = (
                    summary_reward_sums[arm_select]
                    / summary_counts[arm_select]
                )
                if t ==0:
                    all_rewards_sum.append(reward+1)
                    new_rews = all_rewards_sum
                    index = all_rewards_sum.index(reward)
                    loss = l.train(context[arm], new_rews[index]/100)
                else:
                    new_rews = all_rewards_sum
                    new_rews = (new_rews-np.mean(all_rewards_sum))/np.std(all_rewards_sum)
                    index = all_rewards_sum.index(reward)
                    loss = l.train(context[arm], new_rews[index])
        print(f"summary averages: {summary_avg_array}")
        if (t+1) % 5 == 0:
            print('{}: {:.3f}, {:.3f}, {:.3f}'.format(t+1, summ, rewards,loss))
    all_regrets.append(regrets)
    all_rewards.append(rewards_list)
    all_costs.append(costs_list)
    all_costs_summarizer.append(costs_list_summarizer)
    all_costs_debate.append(costs_list_halluc)
    all_plays.append(plays_no.copy())
    all_avg_arrays.append(avg_array.copy())
    all_avg_summary_arrays.append(summary_avg_array.copy()) 


import pickle
import pandas as pd
avg_error = { arm: sum(errs)/len(errs) 
              for arm, errs in cost_error_dict.items()
              if errs }
regrets_arr = np.array(all_regrets)     # shape (no_runs, num_rounds)
rewards_arr = np.array(all_rewards)
costs_arr   = np.array(all_costs)
costs_summarizer_arr   = np.array(all_costs_summarizer)
costs_debate_arr   = np.array(all_costs_debate)
plays_arr   = np.array(all_plays)       # shape (no_runs, num_arms)
avg_df      = pd.DataFrame(all_avg_arrays)  # columns=model names

regrets_mean = regrets_arr.mean(axis=0)
regrets_std  = regrets_arr.std(axis=0)
rewards_mean = rewards_arr.mean(axis=0)
rewards_std  = rewards_arr.std(axis=0)
costs_mean   = costs_arr.mean(axis=0)
costs_std    = costs_arr.std(axis=0)
costs_summarizer_mean   = costs_summarizer_arr.mean(axis=0)
costs_summarizer_std    = costs_summarizer_arr.std(axis=0)
costs_debate_mean   = costs_debate_arr.mean(axis=0)
costs_debate_std    = costs_debate_arr.std(axis=0)
plays_mean = plays_arr.mean(axis=0)
plays_std  = plays_arr.std(axis=0)
avg_mean = avg_df.mean(axis=0).to_dict()
avg_std  = avg_df.std(axis=0).to_dict()

summary_avg_df   = pd.DataFrame(all_avg_summary_arrays)
summary_mean     = summary_avg_df.mean(axis=0).to_dict()
summary_std      = summary_avg_df.std(axis=0).to_dict()

import pickle
pickle.dump(regrets_mean, open("med_results/regrets_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(regrets_std,  open("med_results/regrets_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(rewards_mean, open("med_results/rewards_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(rewards_std,  open("med_results/rewards_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(costs_mean,   open("med_results/costs_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(costs_std,    open("med_results/costs_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(costs_summarizer_mean,   open("med_results/costs_summarizer_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(costs_summarizer_std,    open("med_results/costs_summarizer_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(costs_debate_mean,   open("med_results/costs_debate_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(costs_debate_std,    open("med_results/costs_debate_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(plays_mean,   open("med_results/plays_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(plays_std,    open("med_results/plays_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(avg_mean,     open("med_results/avg_accuracy_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(avg_std,      open("med_results/avg_accuracy_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(summary_mean,     open("med_results/sum_avg_accuracy_mean_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(summary_std,      open("med_results/sum_avg_accuracy_std_med_budgeted_neucb_3.pkl","wb"))
pickle.dump(avg_error,      open("med_results/avg_cost_err_med_budgeted_neucb_3.pkl","wb"))
print(f"Final mean regret: {regrets_mean[-1]}")
print(f"Final mean reward: {rewards_mean[-1]}")
print(f"Final mean cost: {costs_mean[-1]}")
print(f"Final mean summarizer cost: {costs_summarizer_mean[-1]}")
print(f"Final mean plays: {plays_mean}")
print(f"Final mean average array: {avg_mean}")
print(f"Final mean summary average array: {summary_mean}")
print(f"Final mean cost error: {avg_error}")


print("All runs complete. Summary pickles written.")