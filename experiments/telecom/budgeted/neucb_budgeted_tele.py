import numpy as np
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import extend
from src.prompts.prompt_maker import input_maker
from src.embedding.embed_tele import get_context
from utils.helper import opt_eval, get_summary

# ─── STEP 1: Telecom dataset ────────────────────────────────────────────────────
input_reports, labels, explanations = input_maker("seq", "telecom")
dataset = "telecom"

# ─── STEP 2: Description arrays ─────────────────────────────────────────────────
summary_description_array = [
    "Summarize the telecommunications question and its options concisely for analysis.",
    "Provide a brief recap of the telecom question and choices for researchers.",
    "You will take the role of a telecom-specialist summarizer. Summarize the question and answer options.",
    "Produce a short summary of the telecom question and all choices.",
    "Present the telecom question and its multiple-choice options in a concise summary."
]

diagnosis_description_array = [
    "Answer the telecom MCQ strictly 'option {i}' for this question.",
    "Provide the MCQ answer (1–4) for this telecom question.",
    "Output the telecom MCQ response as 'option {i}'.",
    "Select the correct option (1–4) for the telecommunications question.",
    "Choose the telecom MCQ answer and output 'option {i}'."
]

explanation_description_array = [
    "Explain in detail why the chosen telecom MCQ answer is correct.",
    "Provide a step-by-step rationale for why the selected answer is correct.",
    "As a telecom expert, justify why the chosen MCQ option is right.",
    "Offer a clear explanation for why the selected telecom answer is correct.",
    "Give a detailed rationale for why the chosen option is correct."
]

# ─── STEP 3: Deployment instructions per arm ────────────────────────────────────
documents = summary_description_array+ diagnosis_description_array+ explanation_description_array+ list(input_reports)

# ─── STEP 5: Deployment instructions per arm ───────────────────────────────────────────
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

# ─── STEP 4: Cost-per-token dictionaries ────────────────────────────────────────
cost_per_token = {
    "base"            : 0.0000015,
    "assistants"      : 0.0000015,
    "finetune_med"    : 0.00001,
    "finetune_tele"   : 0.00001,
    "finetune_med_new": 0.00001,
    "llama"           : 0.00000071
}

input_cost_per_token = {
    "base"            : 0.0000005,
    "assistants"      : 0.0000005,
    "finetune_med"    : 0.00000025,
    "finetune_tele"   : 0.00000025,
    "finetune_med_new": 0.00000025,
    "llama"           : 0.00000071
}
# ─── STEP 5: Token-length predictor ─────────────────────────────────────────────
from transformers import AutoConfig, AutoTokenizer
import json
from src.token_prediction.tok_length_predict import BertRegressionModel
import tiktoken

reg_model_name = "bert-base-uncased"
reg_config     = AutoConfig.from_pretrained(reg_model_name)
reg_tokenizer  = AutoTokenizer.from_pretrained(reg_model_name)
with open("model_names.json") as f:
    orig_model_names = json.load(f)
num_models = len(orig_model_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token_length_model = BertRegressionModel(
    reg_config, reg_model_name,
    hidden_dim=128,
    num_models=num_models
).to(device)
token_length_model.load_state_dict(torch.load("best_length_model.pth", map_location=device))
token_length_model.eval()

from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

import tiktoken
openai_models = {"gpt-3.5-turbo", "gpt-4"}
encodings = { m: tiktoken.encoding_for_model(m) for m in openai_models }
from transformers import AutoTokenizer as HFTokenizer
try:
    llama_tok = HFTokenizer.from_pretrained("openlm-research/open_llama_13b")
except Exception:
    llama_tok = reg_tokenizer


arm_to_llm = {
        "base"            : "gpt-3.5-turbo",
        "assistants"      : "gpt-3.5-turbo",
        "finetune_med"    : "gpt-4",
        "finetune_tele"   : "gpt-4",
        "finetune_med_new": "gpt-4",
        "llama"           : "llama-13b"
    }

arm_encoders = {}
for mk, llm_name in arm_to_llm.items():
    if llm_name in encodings:
        arm_encoders[mk] = encodings[llm_name]
    else:
        arm_encoders[mk] = llama_tok

# ─── STEP 6: NeuralUCB Bandit ──────────────────────────────────────────────────
class NeuralUCBDiag:
    def __init__(self, style, dim, lamdba=1, nu=1, hidden=100):
        self.device = device
        self.net    = extend(nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden,1)).to(self.device))
        self.lamdba = lamdba
        self.nu     = nu
        p_count     = sum(p.numel() for p in self.net.parameters())
        self.U      = lamdba * torch.ones(p_count, device=self.device)
        self.contexts = []
        self.rewards  = []
        self.style    = style
    def selection(self, context, style):
        x = torch.from_numpy(context).float().to(self.device).unsqueeze(0)
        mu = self.net(x)
        self.net.zero_grad(); mu.backward(retain_graph=True)
        grads = torch.cat([p.grad.flatten() for p in self.net.parameters()])
        sigma = torch.sqrt(torch.sum((self.lamdba*self.nu*grads*grads/self.U)))
        score = (0.2*mu.item()+2*sigma.item()) if style=='ucb' else torch.normal(1.5*mu.view(-1),0.01*sigma.view(-1)).item()
        self.U += grads*grads
        return score
    def train(self, context, reward):
        c = torch.from_numpy(context).float().to(self.device).unsqueeze(0)
        self.contexts.append(c); self.rewards.append(float(reward))
        optimizer = optim.SGD(self.net.parameters(), lr=1e-4, weight_decay=self.lamdba)
        tot_loss = 0; cnt=0
        for ctx, r in zip(self.contexts, self.rewards):
            optimizer.zero_grad()
            pred = self.net(ctx).view(-1)[0]
            loss = (pred-r)**2
            loss.backward(); optimizer.step()
            tot_loss += loss.item(); cnt+=1
            if cnt>=5: break
        return tot_loss/cnt if cnt>0 else 0

# ─── STEP 7: Args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--size', default=100, type=int, help='number of rounds')
parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularization')
parser.add_argument('--hidden', type=int, default=50, help='network hidden size')
parser.add_argument('--style', default='ts', metavar='ts|ucb', help='TS or UCB')
parser.add_argument('--number_tasks', default=3, type=int, help='number of subtasks')
parser.add_argument('--no_runs', default=2, type=int, help='how many independent runs')
parser.add_argument('--alpha', default=125, type=int, help='cost accuracy tradeoff weight')
args = parser.parse_args()
size, nu, lamdba, hidden, style, number_tasks, no_runs, alpha = (
    args.size, args.nu, args.lamdba, args.hidden, args.style,
    args.number_tasks, args.no_runs, args.alpha
)
num_rounds = size

# ─── STEP 8: Prepare models & containers ───────────────────────────────────────
models_summarizer = list(deployments_summarizer.keys())
models_diagnoser  = list(deployments_diagnoser.keys())
models_explainer  = list(deployments_explainer.keys())
all_regrets, all_rewards, all_costs = [], [], []
all_plays_s, all_plays_d, all_plays_e = [], [], []
all_avg_arrays = []


# ─── RUN SIMULATIONS ───────────────────────────────────────────────────────────
for run in range(args.no_runs):
    print(f"=== Run {run+1}/{args.no_runs} ===")
    u_sum = NeuralUCBDiag(args.style, 384, args.lamdba, args.nu, args.hidden)
    u_diag = NeuralUCBDiag(args.style, 384, args.lamdba, args.nu, args.hidden)
    u_exp  = NeuralUCBDiag(args.style, 384, args.lamdba, args.nu, args.hidden)
    plays_s = np.zeros(len(deployments_summarizer), int)
    plays_d = np.zeros(len(deployments_diagnoser), int)
    plays_e = np.zeros(len(deployments_explainer), int)
    regrets, rewards, costs = [], [], []
    tot_reward = 0; cum_reg = 0
    avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"llama":0}
    i = 0
    documents = (
        summary_description_array
        + diagnosis_description_array
        + explanation_description_array
        + list(input_reports)
    )

    all_rewards_sum = []
    all_rewards_debate = []
    all_rewards_diag = []

    arm_to_llm = {
        "base"            : "gpt-3.5-turbo",
        "assistants"      : "gpt-3.5-turbo",
        "finetune_med"    : "gpt-4",
        "finetune_tele"   : "gpt-4",
        "finetune_med_new": "gpt-4",
        "llama"           : "llama-13b"
    }

    for t in range(args.size):
        print(f"Round {t+1}")
        # --- Summarization Subtask ---
        models = list(deployments_summarizer.keys())
        prompt_s = input_reports[t]
        feats_s = [
            get_context(
                documents, t, i, j,
                len(summary_description_array),
                len(diagnosis_description_array),
                len(input_reports), inp_model, dataset
            )
            for j in range(len(summary_description_array))
        ]
        raw_s = [u_sum.selection(ctx, args.style) for ctx in feats_s]
        # Cost prediction & penalization
        pred_s, in_s = [], []
        toks = reg_tokenizer(
            text=prompt_s, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt"
        )
        for mk in models:
            llm = arm_to_llm[mk]
            # Input cost
            enc = arm_encoders[mk]
            if hasattr(enc, 'encode'):
                in_len = len(enc.encode(prompt_s))
            else:
                in_len = len(enc(prompt_s, truncation=True)["input_ids"])
            in_s.append(in_len)
            # Output prediction cost
            idx = orig_model_names.index(llm)
            onehot = torch.zeros(num_models, device=device)
            onehot[idx] = 1.0
            with torch.no_grad():
                pred = token_length_model(
                    toks["input_ids"], toks["attention_mask"], onehot.unsqueeze(0)
                )
            pred_s.append(pred.item())
        pen_s = [
            raw_val - args.alpha * (
                input_cost_per_token[mk] * in_s[k]
                + cost_per_token[mk]     * pred_s[k]
            )
            for (raw_val, mk, k) in zip(raw_s, models, range(len(raw_s)))
        ]
        i_s = int(np.argmax(pen_s)); arm_s = models[i_s]
        print(f"Selected summarizer: {arm_s}")
        # Generate summary
        summary = get_summary(input_reports[t], arm_s, "tele")
        cost_s = (
            input_cost_per_token[arm_s] * in_s[i_s]
            + cost_per_token[arm_s] * pred_s[i_s]
        )
        u_sum.train(feats_s[i_s], 0.0)
        plays_s[i_s] += 1

        # --- Diagnosis Subtask ---
        models  = ["base", "finetune_med", "finetune_tele", "finetune_med_new", "llama"]
        prompt_d = summary
        prompt_d = prompt_d + "Please give the correct option in the format: option [correct option number]."
        prompt_d = prompt_d.replace('\n','')
        models_d = list(deployments_diagnoser.keys())
        feats_d = [
            get_context(
                documents, t, i, len(summary_description_array) + j,
                len(summary_description_array),
                len(diagnosis_description_array),
                len(input_reports), inp_model, dataset
            )
            for j in range(len(diagnosis_description_array))
        ]
        raw_d, pred_d, in_d = [], [], []
        raw_d = [u_diag.selection(ctx, args.style) for ctx in feats_d]
        toks_d = reg_tokenizer(
            prompt_d, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt"
        ).to(device)
        for mk in models:
            llm = arm_to_llm[mk]
            enc = arm_encoders[mk]
            in_len = len(enc.encode(prompt_d)) if hasattr(enc, 'encode') else len(enc(prompt_d, truncation=True)["input_ids"])
            in_d.append(in_len)
            idx = orig_model_names.index(llm)
            onehot = torch.zeros(num_models, device=device); onehot[idx] = 1.0
            with torch.no_grad():
                pred = token_length_model(
                    toks_d["input_ids"], toks_d["attention_mask"], onehot.unsqueeze(0)
                )
            pred_d.append(pred.item())
        pen_d = [
            raw_d[k] - args.alpha * (
                input_cost_per_token[mk] * in_d[k] + cost_per_token[mk] * pred_d[k]
            )
            for k, mk in enumerate(models_d)
        ]
        i_d = int(np.argmax(pen_d)); arm_d = models[i_d]
        print(f"Selected diagnoser: {arm_d}")
        reg1, reward1, out_len1, avg_array, _, _ = opt_eval(
            deployments_diagnoser, prompt_d, "diagnosis",
            arm_d, avg_array, t, all_rewards_sum, all_rewards_diag, labels, dataset
        )
        cost_d = (input_cost_per_token[arm_d] * in_d[i_d] + cost_per_token[arm_d] * out_len1)
        u_diag.train(feats_d[i_d], reward1)
        plays_d[i_d] += 1
        cum_reg += reg1
        tot_reward += reward1

        # --- Explanation Subtask ---
        models  = ["base", "finetune_med", "finetune_tele", "finetune_med_new", "llama"]
        prompt_e = input_reports[t] + "Answer chosen: " + str(reward1)
        models_e = list(deployments_explainer.keys())
        feats_e = [
            get_context(
                documents, t, i,
                len(summary_description_array) + len(diagnosis_description_array) + j,
                len(summary_description_array),
                len(diagnosis_description_array),
                len(input_reports), inp_model, dataset
            )
            for j in range(len(explanation_description_array))
        ]
        raw_e, pred_e, in_e = [], [], []
        raw_e = [u_exp.selection(ctx, args.style) for ctx in feats_e]
        toks_e = reg_tokenizer(
            text=prompt_e, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt"
        ).to(device)
        for mk in models:
            llm = arm_to_llm[mk]
            enc = arm_encoders[mk]
            in_len = len(enc.encode(prompt_e)) if hasattr(enc, 'encode') else len(enc(prompt_e, truncation=True)["input_ids"])
            in_e.append(in_len)
            idx = orig_model_names.index(llm)
            onehot = torch.zeros(num_models, device=device); onehot[idx] = 1.0
            with torch.no_grad():
                pred = token_length_model(
                    toks_e["input_ids"], toks_e["attention_mask"], onehot.unsqueeze(0)
                )
            pred_e.append(pred.item())
        pen_e = [
            raw_e[k] - args.alpha * (
                input_cost_per_token[mk] * in_e[k] + cost_per_token[mk] * pred_e[k]
            )
            for k, mk in enumerate(models_e)
        ]
        i_e = int(np.argmax(pen_e)); arm_e = models[i_e]
        print(f"Selected explainer: {arm_e}")
        reg2, reward2, out_len2, avg_array, _, _ = opt_eval(
            deployments_explainer, prompt_e, "explanation",
            arm_e, avg_array, t, all_rewards_sum, all_rewards_diag, explanations, dataset
        )
        cost_e = (input_cost_per_token[arm_e] * in_e[i_e] + cost_per_token[arm_e] * out_len2)
        u_exp.train(feats_e[i_e], reward2)
        plays_e[i_e] += 1
        tot_reward += reward2

        # Record metrics
        cum_reg += reg2
        regrets.append(cum_reg)
        rewards.append(tot_reward)
        costs.append(cost_s + cost_d + cost_e)
    
    all_regrets.append(regrets)
    all_rewards.append(rewards)
    all_costs.append(costs)
    all_plays_s.append(plays_s)
    all_plays_d.append(plays_d)
    all_plays_e.append(plays_e)
    all_avg_arrays.append(avg_array.copy())

import pandas as pd
avg_df      = pd.DataFrame(all_avg_arrays)
avg_mean = avg_df.mean(axis=0).to_dict()
avg_std  = avg_df.std(axis=0).to_dict()

# ─── STEP 9: Save metrics ───────────────────────────────────────────────────────
pickle.dump(np.mean(all_regrets,axis=0),open("tele_results/regrets_mean_neucb_budgeted.pkl","wb"))
pickle.dump(np.std(all_regrets,axis=0), open("tele_results/regrets_std_neucb_budgeted.pkl","wb"))
pickle.dump(np.mean(all_rewards,axis=0),open("tele_results/rewards_mean_neucb_budgeted.pkl","wb"))
pickle.dump(np.std(all_rewards,axis=0), open("tele_results/rewards_std_neucb_budgeted.pkl","wb"))
pickle.dump(np.mean(all_costs,axis=0),   open("tele_results/costs_mean_neucb_budgeted.pkl","wb"))
pickle.dump(np.std(all_costs,axis=0),    open("tele_results/costs_std_neucb_budgeted.pkl","wb"))
pickle.dump(all_plays_s,open("tele_results/plays_s_neucb_budgeted.pkl","wb"))
pickle.dump(all_plays_d,open("tele_results/plays_d_neucb_budgeted.pkl","wb"))
pickle.dump(all_plays_e,open("tele_results/plays_e_neucb_budgeted.pkl","wb"))
pickle.dump(avg_mean,     open("tele_results/avg_accuracy_mean_tele_budgeted_neucb.pkl","wb"))
pickle.dump(avg_std,      open("tele_results/avg_accuracy_std_tele_budgeted_neucb.pkl","wb"))

print("All runs complete. Summary pickles written.")
