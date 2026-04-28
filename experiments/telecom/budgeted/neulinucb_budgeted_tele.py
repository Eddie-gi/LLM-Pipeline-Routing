import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
from transformers import AutoConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.prompts.prompt_maker import input_maker
from src.token_prediction.tok_length_predict import BertRegressionModel
from src.embedding.embed_tele import get_context

from utils.helper import opt_eval, get_summary

# 1) Regret helper
def get_regret(deployments, prompt, task, selected, avg_array,
               t, all_sum, all_sub, labels, dataset):
    return opt_eval(deployments, prompt, task,
                    selected, avg_array, t,
                    all_sum, all_sub,
                    labels, dataset)

# 2) Length predictor setup
reg_config    = AutoConfig.from_pretrained("bert-base-uncased")
reg_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
with open("model_names.json") as f:
    orig_model_names = json.load(f)
num_models = len(orig_model_names)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token_length_model = BertRegressionModel(
    reg_config, "bert-base-uncased",
    hidden_dim=128, num_models=num_models
).to(device)
token_length_model.load_state_dict(
    torch.load("best_length_model.pth", map_location=device)
)
token_length_model.eval()

# 3) Costs and arm mapping
cost_per_token = {"base":1.5e-6, "assistants":1.5e-6,
                  "finetune_med":1e-5, "finetune_tele":1e-5,
                  "finetune_med_new":1e-5, "llama":7.1e-7}
input_cost_per_token = {"base":5e-7, "assistants":5e-7,
                        "finetune_med":2.5e-7, "finetune_tele":2.5e-7,
                        "finetune_med_new":2.5e-7, "llama":7.1e-7}
arm_to_llm = {"base":"gpt-3.5-turbo","assistants":"gpt-3.5-turbo",
              "finetune_med":"gpt-4","finetune_tele":"gpt-4",
              "finetune_med_new":"gpt-4","llama":"llama-13b"}

# 4) Sherman–Morrison
def inv_sherman_morrison(u, A_inv):
    Au = A_inv.dot(u)
    return A_inv - np.outer(Au,Au)/(1.0+u.dot(Au))

# 5) NeuralLinearUCB
class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, dim)
    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))

class NeuralLinearUCB:
    def __init__(self, dim, lamdba=1, hidden=100, n_arm=5):
        self.n_arm = n_arm
        self.func  = Network(dim, hidden).to(device)
        self.lamdba= lamdba
        self.theta = np.zeros((n_arm, dim))
        self.b     = np.zeros((n_arm, dim))
        self.A_inv = np.array([np.eye(dim) for _ in range(n_arm)])
        self.contexts, self.arms, self.rewards = [], [], []

    def select(self, contexts, pred_lens, models, in_lens, alpha):
        feats = [self.func(torch.from_numpy(c).float().to(device))
                 .cpu().detach().numpy() for c in contexts]
        ucb   = [np.sqrt(f.dot(self.A_inv[i]).dot(f)) for i,f in enumerate(feats)]
        mu    = [f.dot(self.theta[i]) for i,f in enumerate(feats)]
        scores = [mu[i] + alpha*ucb[i]
                  - alpha*(cost_per_token[m]*pred_lens[i]
                           + input_cost_per_token[m]*in_lens[i])
                  for i,m in enumerate(models)]
        return int(np.argmax(scores))

    def train(self, context, arm, reward):
        self.contexts.append(torch.from_numpy(context.reshape(1,-1)).float())
        self.arms.append(arm)
        self.rewards.append(reward)
        opt = optim.SGD(self.func.parameters(), lr=1e-3,
                        weight_decay=self.lamdba)
        loss_val=0.0
        for _ in range(5):
            for c,a,r in zip(self.contexts,self.arms,self.rewards):
                opt.zero_grad()
                feat = self.func(c.to(device))
                mu   = (feat * torch.from_numpy(self.theta[a])
                         .float().to(device)).sum()
                loss = (mu - r)**2
                loss.backward()
                opt.step()
                loss_val = loss.item()
        return loss_val

    def update_model(self, context, arm, reward):
        f = self.func(torch.from_numpy(context).float().to(device))
        f = f.cpu().detach().numpy()
        self.b[arm] += f * reward
        self.A_inv[arm] = inv_sherman_morrison(f, self.A_inv[arm])
        self.theta = np.array([self.A_inv[i].dot(self.b[i])
                               for i in range(self.n_arm)])

# 6) Subtask descriptions
summary_desc = ["Summarize the telecom question and options.",
                "Briefly summarize question + MCQ.",
                "Telecom specialist: summarize.",
                "Concise telecom summary.",
                "Short telecom question recap."]
diag_desc = ["MCQ telecom: output option [1-4].",
             "Telecom MCQ specialist: option {i}.",
             "General LLM: choose option.",
             "Pick telecom MCQ answer.",
             "Select option 1-4."]
expl_desc = ["Explain why answer is correct.",
             "Provide rationale for choice.",
             "Step-by-step explanation.",
             "Detailed telecom explanation.",
             "Why option {i} is correct."]
arms = list(arm_to_llm.keys())

from collections import OrderedDict
deployments_summarizer = {
    "base"            : ("gpt-35-turbo", "You are to summarize a telecom question and its options."),
    "assistants"      : ("Assistant",     "You are to summarize a telecom question and its options."),
    "finetune_med"    : ("Med",           "You are to summarize a telecom question and its options."),
    "finetune_tele"   : ("Tele",          "You are to summarize a telecom question and its options."),
    "finetune_med_new": ("Med_New",       "You are to summarize a telecom question and its options."),
    "llama"           : ("llama",         "You are to summarize a telecom question and its options."),
}

deployments_diagnoser = {
    "base"            : ("gpt-35-turbo",
                         "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}' where i∈{1,2,3,4}."),
    "finetune_med"    : ("Med",
                         "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'."),
    "finetune_tele"   : ("Tele",
                         "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'."),
    "finetune_med_new": ("Med_New",
                         "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'."),
    "llama"           : ("llama",
                         "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'.")
}

deployments_explainer = {
    "base"            : ("gpt-35-turbo", 
                         "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "finetune_med"    : ("Med", 
                         "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "finetune_tele"   : ("Tele", 
                         "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "finetune_med_new": ("Med_New", 
                         "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "llama"           : ("llama", 
                         "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale.")
}

# 7) Main loop
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=100)
    parser.add_argument('--hidden',type=int,default=50)
    parser.add_argument('--alpha',type=float,default=125)
    parser.add_argument('--no_runs',type=int,default=1)
    args=parser.parse_args()

    input_reports,labels,explanations=input_maker('seq','telecom')
    emb_model=SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # encoders
    import tiktoken
    encodings={m:tiktoken.encoding_for_model(m) for m in ['gpt-3.5-turbo','gpt-4']}
    from transformers import AutoTokenizer as HTok
    try: llama_tok=HTok.from_pretrained('openlm-research/open_llama_13b')
    except: llama_tok=reg_tokenizer
    arm_enc={a:(encodings[arm_to_llm[a]] if arm_to_llm[a] in encodings else llama_tok) for a in arms}

    all_regrets,all_rewards,all_costs,all_plays,all_avg=[],[],[],[],[]
    all_plays_s, all_plays_d, all_plays_e = [], [], []
    for run in range(args.no_runs):
        print(f"Starting run {run+1}/{args.no_runs}")
        bs=NeuralLinearUCB(384,1,args.hidden,len(arms))
        bd=NeuralLinearUCB(384,1,args.hidden,len(arms))
        be=NeuralLinearUCB(384,1,args.hidden,len(arms))
        plays_s = np.zeros(len(deployments_summarizer), int)
        plays_d = np.zeros(len(deployments_diagnoser), int)
        plays_e = np.zeros(len(deployments_explainer), int)
        plays=np.zeros(len(arms)*3)
        avg_acc = {
            "gpt-35-turbo": 0.0,
            "Med"         : 0.0,
            "Tele"        : 0.0,
            "Med_New"     : 0.0,
            "llama"       : 0.0
        }
        regrets,rews,costs=[],[],[]
        cum_reg=tot_r=tot_c=0.0
        all_rewards_diag=[]

        for t in range(args.size):
            # Summary subtask
            ctx0=[get_context(summary_desc+diag_desc+expl_desc+list(input_reports),
                              t,0,i,len(summary_desc),len(diag_desc),len(input_reports),emb_model,'telecom')
                  for i in range(len(arms))]
            prompt0=input_reports[t]
            toks0=reg_tokenizer(prompt0, return_tensors='pt', padding='max_length', truncation=True, max_length=256).to(device)
            pred0=[token_length_model(
                        toks0['input_ids'],
                        toks0['attention_mask'],
                        torch.eye(num_models)[orig_model_names.index(arm_to_llm[a])].unsqueeze(0).to(device)
                   ).item() for a in arms]
            in0=[len(arm_enc[a].encode(prompt0)) for a in arms]
            i0=bs.select(ctx0,pred0,arms,in0,args.alpha)
            print(f"Run {run+1}, Round {t+1}: Summary selected -> {arms[i0]}")
            summary=get_summary(prompt0,arms[i0],'tele')
            bs.update_model(np.array(ctx0[i0]),i0,0)
            tot_c += input_cost_per_token[arms[i0]] * in0[i0] + cost_per_token[arms[i0]] * pred0[i0]
            plays[i0]+=1
            plays_s[i0] += 1

            # Diagnosis subtask
            prompt1=f"{prompt0}\nSummary: {summary}\nChoose [1-4]."
            ctx1=[get_context(summary_desc+diag_desc+expl_desc+list(input_reports)+[summary],
                              t,1,i,len(summary_desc),len(diag_desc),len(input_reports),emb_model,'telecom')
                  for i in range(len(arms))]
            toks1=reg_tokenizer(prompt1,return_tensors='pt',padding='max_length', truncation=True, max_length=256).to(device)
            pred1=[token_length_model(
                        toks1['input_ids'],
                        toks1['attention_mask'],
                        torch.eye(num_models)[orig_model_names.index(arm_to_llm[a])].unsqueeze(0).to(device)
                   ).item() for a in arms]
            in1=[len(arm_enc[a].encode(prompt1)) for a in arms]
            diag_arms = list(deployments_diagnoser.keys())  # ["base","finetune_med",…,"llama"]
            i1 = bd.select(ctx1, pred1, diag_arms, in1, args.alpha)
            selected = diag_arms[i1]

            print(f"Run {run+1}, Round {t+1}: Diagnosis selected -> {selected}")
            reg1,r1,out_len1,avg_acc,_,_ = get_regret(deployments_diagnoser,prompt1,'diagnosis',selected,avg_acc,t,[],[],labels,'telecom')
            bd.update_model(np.array(ctx1[i1]),i1,r1)
            plays[len(arms)+i1]+=1; tot_r+=r1
            plays_d[i1] += 1
            
            tot_c += input_cost_per_token[selected] * in1[i1] + cost_per_token[selected] * out_len1

            # Diagnosis training
            all_rewards_diag.append(0 if r1==0 else 1)
            arr=np.array(all_rewards_diag)
            norm=(arr-arr.mean())/arr.std() if arr.std()>0 else arr
            idx=len(all_rewards_diag)-1
            loss_d=bd.train(ctx1[i1],i1,float(norm[idx]))
            if (t+1)%5==0:
                print(f"{t+1}: Diagnosis train loss: {loss_d:.3f}")

            # Explanation subtask
            prompt2=f"{prompt0}\nAnswer: {r1}\nExplain why."
            ctx2=[get_context(summary_desc+diag_desc+expl_desc+list(input_reports)+[summary,prompt2],
                              t,2,i,len(summary_desc),len(diag_desc),len(input_reports),emb_model,'telecom')
                  for i in range(len(arms))]
            expl_arms = list(deployments_explainer.keys())
            in2=[len(arm_enc[a].encode(prompt2)) for a in arms]
            i2 = be.select(ctx2, [0]*len(expl_arms), expl_arms, [0]*len(expl_arms), args.alpha)
            selected2 = expl_arms[i2]
            print(f"Run {run+1}, Round {t+1}: Explanation selected -> {selected2}")
            reg2,r2,out_len2,avg_acc,_,_ = get_regret(deployments_explainer,prompt2,'explanation',selected2,avg_acc,t,[],[],explanations,'telecom')
            be.update_model(np.array(ctx2[i2]),i2,r2)
            tot_r+=r2; cum_reg+=(reg1+reg2)
            tot_c += input_cost_per_token[selected2] * in2[i2] + cost_per_token[selected2] * out_len2
            regrets.append(cum_reg); rews.append(tot_r); costs.append(tot_c)
            plays[len(arms)+i1]+=1
            plays_e[i2] += 1

        print(f"Run {run+1} complete: Final regret = {cum_reg:.4f}")
        all_regrets.append(regrets); all_rewards.append(rews)
        all_costs.append(costs); all_plays.append(plays)
        all_avg.append(avg_acc)
        all_plays_s.append(plays_s)
        all_plays_d.append(plays_d)
        all_plays_e.append(plays_e)

    # Final aggregation & pickles
    print("Aggregating results and writing pickles...")
    R=np.array(all_regrets); W=np.array(all_rewards)
    C=np.array(all_costs); P=np.array(all_plays)
    pickle.dump(R.mean(0),open("regrets_mean_3sub.pkl","wb"))
    pickle.dump(R.std(0),open("regrets_std_3sub.pkl","wb"))
    pickle.dump(W.mean(0),open("rewards_mean_3sub.pkl","wb"))
    pickle.dump(W.std(0),open("rewards_std_3sub.pkl","wb"))
    pickle.dump(C.mean(0),open("costs_mean_3sub.pkl","wb"))
    pickle.dump(C.std(0),open("costs_std_3sub.pkl","wb"))
    pickle.dump(P.mean(0),open("plays_mean_3sub.pkl","wb"))
    pickle.dump(P.std(0),open("plays_std_3sub.pkl","wb"))
    pickle.dump(all_avg,open("avg_array_3sub.pkl","wb"))
    pickle.dump(all_plays_s,open("tele_results/plays_s_neulinucb_budgeted.pkl","wb"))
    pickle.dump(all_plays_d,open("tele_results/plays_d_neulinucb_budgeted.pkl","wb"))
    pickle.dump(all_plays_e,open("tele_results/plays_e_neulinucb_budgeted.pkl","wb"))
    print(f"Mean regret: {R.mean(0)[-1]:.4f}")
    print("All results loaded")
