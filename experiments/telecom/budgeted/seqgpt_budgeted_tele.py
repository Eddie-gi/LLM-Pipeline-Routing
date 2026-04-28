import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scipy as sp
import argparse
from backpack import extend
from backpack.extensions import BatchGrad
from transformers import AutoConfig, AutoTokenizer
from src.embedding.embed_tele import get_context
from src.prompts.prompt_maker import input_maker
from src.regrets.final_rand_med   import final_eval
from utils.helper import opt_eval, get_summary

def get_regret(deployments, prompt, task, selected, avg_array, t,
               all_rewards_sum, all_rewards_sub, labels, dataset):
    return opt_eval(deployments, prompt, task,
                    selected, avg_array, t,
                    all_rewards_sum, all_rewards_sub,
                    labels, dataset)

def get_reward(deployment, cat, prompt, task,
               all_rewards_sum, all_rewards_sub, summary):
    return final_eval(deployment, cat, prompt,
                      task, all_rewards_sum, all_rewards_sub, summary)

reg_model_name  = "bert-base-uncased"
reg_config      = AutoConfig.from_pretrained(reg_model_name)
reg_tokenizer   = AutoTokenizer.from_pretrained(reg_model_name)

import json
with open("model_names.json") as f:
    orig_model_names = json.load(f)

num_models = len(orig_model_names)
device     = "cuda" if torch.cuda.is_available() else "cpu"

from src.token_prediction.tok_length_predict import BertRegressionModel
token_length_model = BertRegressionModel(
    reg_config, reg_model_name,
    hidden_dim=128,
    num_models=num_models
).to(device)
token_length_model.load_state_dict(
    torch.load("best_length_model.pth", map_location=device)
)
token_length_model.eval()

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

dataset = "telecom"

input_reports, labels, explinations = input_maker("seq", "telecom")

summary_description_array = [
    "Summarize the telecommunications question and its options concisely for analysis.",
    "Provide a brief recap of the telecom question and choices for researchers.",
    "You will take the role of a telecom‐specialist summarizer. Summarize the question and answer options.",
    "Produce a short summary of the telecom question and all choices.",
    "Present the telecom question and its multiple‐choice options in a concise summary.",
    "You will summarize the telecom question text and options for quick reference."
]

diagnosis_description_array = [
    "General LLM that does not specialize. Output the MCQ answer option (1–4) for this telecom question.",
    "LLM specializing in medical diagnosis. (Ignore—still use for telecom MCQ; output 1–4.)",
    "LLM specializing in telecommunications MCQs. Output strictly 'option {i}' (i = 1–4).",
    "LLM specializing in medical diagnosis. (Ignore—still use for telecom MCQ; output 1–4.)",
    "General LLM that does not specialize. Output strictly 'option {i}' for the telecom MCQ."
]

explanation_description_array = [
    "You are a telecom expert. Explain in detail why the chosen answer to this telecom question is correct.",
    "You are a telecom specialist. Provide a clear rationale for the chosen MCQ answer.",
    "You are a telecom‐domain expert. Give a step-by-step explanation of why your answer is correct.",
    "You are a telecom analyst. Explain thoroughly why the selected answer is correct.",
    "You are a telecom instructor. Provide a detailed explanation of why the chosen MCQ answer is correct."
]

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

class Network(nn.Module):
    def __init__(self, dim=100, hidden_size=100):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

class NeuralUCBDiag:
    def __init__(self, style, dim, lamdba=1, nu=1, hidden=100):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.func = extend(Network(dim, hidden_size=hidden).cuda().model)
        else:
            self.func = Network(dim, hidden_size=hidden).model

        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.nu = nu

        # U‐vector for UCB
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        if self.device == "cuda":
            self.U = lamdba * torch.ones((self.total_param,), device="cuda")
        else:
            self.U = lamdba * torch.ones((self.total_param,))

        # Loss → MSE
        if self.device == "cuda":
            self.loss_func = extend(nn.MSELoss().cuda())
        else:
            self.loss_func = extend(nn.MSELoss())

        self.style = style

    def selection(self, context, style):
        tensor = torch.from_numpy(np.array(context)).float().to(self.device)
        mu     = self.func(tensor)
        self.func.zero_grad()
        mu.backward(retain_graph=True)

        grads  = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
        sigma2 = self.lamdba * self.nu * grads * grads / self.U
        sigma  = torch.sqrt(torch.sum(sigma2))

        if style == "ucb":
            sample_r = mu.item() + 0.5 * sigma.item()
        else:
            std = 0.2 * sigma
            std_val = max(std.item(), 0.0)
            noise = torch.randn(1).item()
            sample_r = mu.item() + noise * std_val

        self.U += grads * grads
        return sample_r

    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length    = len(self.reward)
        index     = np.arange(length)
        np.random.shuffle(index)

        cnt      = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx].to(self.device)
                r = torch.tensor(float(self.reward[idx]), device=self.device)
                optimizer.zero_grad()
                delta = self.func(c) - r
                loss  = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss   += loss.item()
                cnt       += 1
                if cnt >= 5:
                    return tot_loss / 5

            if batch_loss / length <= 1e-3:
                return batch_loss / length

parser = argparse.ArgumentParser(description="NeuralUCB (telecom, 3 subtasks)")
parser.add_argument("--size",          default=100,  type=int,    help="number of rounds")
parser.add_argument("--nu",            default=1.0,  type=float,  help="nu for control variance")
parser.add_argument("--lamdba",        default=1.0,  type=float,  help="lambda for regularization")
parser.add_argument("--hidden",        default=50,   type=int,    help="network hidden size")
parser.add_argument("--style",         default="ts", type=str,    help="ts or ucb")
parser.add_argument("--number_tasks",  default=3,    type=int,    help="now set to 3")
parser.add_argument("--no_runs",       default=1,    type=int,    help="how many independent runs")
parser.add_argument("--alpha",         default=125,  type=int,    help="cost‐accuracy weight (unused?)")
args, unknown = parser.parse_known_args()

no_tasks = args.number_tasks
no_runs  = args.no_runs
alpha    = args.alpha

all_regrets            = []
all_rewards            = []
all_costs              = []
all_costs_summarizer   = []
all_plays              = []
all_avg_arrays         = []
all_avg_summary_arrays = []
all_diag_avg_arrays    = []
all_expl_avg_arrays    = []

class DynamicVariables:
    pass

from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

emb_size  = 384
total_len = (
    len(summary_description_array)
    + len(diagnosis_description_array)
    + len(explanation_description_array)
    + len(input_reports)
)

for run in range(no_runs):
    print(f"\n===== Starting run {run+1}/{no_runs} =====")
    regrets               = []
    rewards               = []
    costs_list            = []
    costs_list_summarizer = []
    cost_summarizer       = 0.0
    plays_no              = np.ones(
        len(summary_description_array)
        + len(diagnosis_description_array)
        + len(explanation_description_array)
    )
    avg_array             = {
        "gpt-35-turbo": 0.0,
        "Med"         : 0.0,
        "Tele"        : 0.0,
        "Med_New"     : 0.0,
        "llama"       : 0.0
    }
    summary_reward_sums = { arm: 0.0 for arm in deployments_summarizer.keys() }
    summary_counts      = { arm: 0   for arm in deployments_summarizer.keys() }
    summary_avg_array   = { arm: 0.0 for arm in deployments_summarizer.keys() }

    diag_reward_sums = { arm: 0.0 for arm in deployments_diagnoser }
    diag_counts      = { arm: 0   for arm in deployments_diagnoser }
    diag_avg_array   = { arm: 0.0 for arm in deployments_diagnoser }

    expl_reward_sums = { arm: 0.0 for arm in deployments_explainer }
    expl_counts      = { arm: 0   for arm in deployments_explainer }
    expl_avg_array   = { arm: 0.0 for arm in deployments_explainer }

    all_rewards_sum  = []
    all_rewards_diag = []
    all_rewards_expl = []

    dyn_vars = DynamicVariables()
    for i in range(
        len(summary_description_array)
        + len(diagnosis_description_array)
        + len(explanation_description_array)
    ):
        setattr(dyn_vars, f"l_{i}", NeuralUCBDiag(
            args.style, emb_size,
            args.lamdba, args.nu,
            args.hidden
        ))

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

    deploy = [deployments_summarizer, deployments_diagnoser, deployments_explainer]

    tot_reward = 0.0
    costs   = 0.0

    for t in range(args.size):
        # subtask summary
        context = []
        values  = []
        models  = [
            "assistants", "base", "finetune_med",
            "finetune_tele", "finetune_med_new", "llama"
        ]
        prompt_to_model = input_reports[t].replace("\n", " ")
        task            = "summary"

        # Build UCB contexts & values
        for j in range(len(summary_description_array)):
            l = getattr(dyn_vars, f"l_{j}")
            cont = get_context(
                summary_description_array
                + diagnosis_description_array
                + explanation_description_array
                + list(input_reports),
                t,  # round
                0,  # subtask 0
                j,  # context index
                len(summary_description_array),
                len(diagnosis_description_array),
                len(input_reports),
                inp_model,
                dataset
            )
            context.append(cont)
            values.append(l.selection(cont, args.style))

        # Token‐length prediction
        toks = reg_tokenizer(
            prompt_to_model,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(device)

        pred_lengths = []
        for mk in models:
            llm_name = arm_to_llm[mk]
            idx      = orig_model_names.index(llm_name)
            onehot   = torch.zeros(len(orig_model_names), device=device)
            onehot[idx] = 1.0
            onehot    = onehot.unsqueeze(0)
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
            if hasattr(enc, "encode"):
                in_len = len(enc.encode(prompt_to_model))
            else:
                in_len = len(enc(prompt_to_model, truncation=True)["input_ids"])
            in_lengths.append(in_len)

        values_f = [float(v) for v in values]
        print(f"Round {t+1}  |>  UCB values (summarizer): {values_f}")

        cost = []
        for i, mk in enumerate(models):
            values[i] = (
                values[i]- alpha*cost_per_token[mk]* pred_lengths[i]- alpha*input_cost_per_token[mk] * in_lengths[i]
            )
            cost.append(alpha*cost_per_token[mk]* pred_lengths[i] + alpha*input_cost_per_token[mk] * in_lengths[i])
        print(f"Budget cost: {cost}")

        if args.style == "ts":
            values_np = [
                v.cpu().item() if isinstance(v, torch.Tensor) else v
                for v in values
            ]
            arm_idx = np.random.choice(
                np.where(np.array(values_np) == np.array(values_np).max())[0]
            )
        else:
            arm_idx = int(np.argmax(values))

        plays_no[arm_idx] += 1
        arm_select_summarizer = models[arm_idx]
        print(f"Selected summarizer: {arm_select_summarizer}")

        summary = get_summary(input_reports[t], arm_select_summarizer, "tele")

        prompt_to_model = input_reports[t] + "\n\nSummary:\n" + summary
        print(f"Prompt→summ‐evaluator: {prompt_to_model[:80]}...")

        reward = 0
        out_len = len(summary.split())

        all_rewards_sum.append(reward)

        rews_array     = np.array(all_rewards_sum, dtype=float)
        mean_rew       = rews_array.mean()
        std_rew        = rews_array.std()
        if std_rew == 0:
            idx0     = 0
            train_rew = rews_array[0]
        else:
            normalized     = (rews_array - mean_rew) / std_rew
            normalized_val = (reward - mean_rew) / std_rew
            idx0           = normalized.tolist().index(float(normalized_val))
            train_rew      = normalized_val

        l0 = getattr(dyn_vars, f"l_{arm_idx}")
        loss = l0.train(context[arm_idx], train_rew)

        cost_summarizer += (
            input_cost_per_token[arm_select_summarizer] * in_lengths[arm_idx]
            + cost_per_token[arm_select_summarizer] * out_len
        )
        costs_list_summarizer.append(cost_summarizer)

        costs += (
            input_cost_per_token[arm_select_summarizer] * in_lengths[arm_idx]
            + cost_per_token[arm_select_summarizer] * out_len
        )

        summary_reward_sums[arm_select_summarizer] += reward
        summary_counts[arm_select_summarizer]      += 1
        summary_avg_array[arm_select_summarizer]   = (
            summary_reward_sums[arm_select_summarizer]
            / summary_counts[arm_select_summarizer]
        )

        prompt_to_model = prompt_to_model + "Please give the correct option in the format: option [correct option number]."


        #subtask solver
        context = []
        values  = []
        models  = ["base", "finetune_med", "finetune_tele", "finetune_med_new", "llama"]
        task    = "diagnosis"

        documents = (
            summary_description_array
            + diagnosis_description_array
            + explanation_description_array
            + list(input_reports)
            + [summary]
        )

        for j in range(len(diagnosis_description_array)):
            l = getattr(
                dyn_vars, 
                f"l_{len(summary_description_array) + j}"
            )
            cont = get_context(
                documents,
                t,
                1,  # subtask 1
                j,
                len(summary_description_array),
                len(diagnosis_description_array),
                len(input_reports),
                inp_model,
                dataset
            )
            context.append(cont)
            values.append(l.selection(cont, args.style))

        toks = reg_tokenizer(
            prompt_to_model,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(device)

        pred_lengths = []
        for mk in models:
            llm_name = arm_to_llm[mk]
            idx      = orig_model_names.index(llm_name)
            onehot   = torch.zeros(len(orig_model_names), device=device)
            onehot[idx] = 1.0
            onehot    = onehot.unsqueeze(0)
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
            if hasattr(enc, "encode"):
                in_len = len(enc.encode(prompt_to_model))
            else:
                in_len = len(enc(prompt_to_model, truncation=True)["input_ids"])
            in_lengths.append(in_len)

        values_f = [float(v) for v in values]
        print(f"Accuracy UCB (diagnoser): {values_f}")

        cost = []
        for i, mk in enumerate(models):
            values[i] = (
                values[i]- alpha*cost_per_token[mk]* pred_lengths[i]- alpha*input_cost_per_token[mk] * in_lengths[i]
            )
            cost.append(alpha*cost_per_token[mk]* pred_lengths[i] + alpha*input_cost_per_token[mk] * in_lengths[i])
        print(f"Budget cost: {cost}")

        if args.style == "ts":
            values_np = [
                v.cpu().item() if isinstance(v, torch.Tensor) else v
                for v in values
            ]
            arm_idx_1 = np.random.choice(
                np.where(np.array(values_np) == np.array(values_np).max())[0]
            )
        else:
            arm_idx_1 = int(np.argmax(values))

        plays_no[len(summary_description_array) + arm_idx_1] += 1
        arm_select_diagnoser = models[arm_idx_1]
        print(f"Selected solver: {arm_select_diagnoser}")

        if arm_select_diagnoser in ["finetune_med", "finetune_tele", "finetune_med_new"]:
            cat = "finetune"
        else:
            cat = arm_select_diagnoser

        deployment = deployments_diagnoser[arm_select_diagnoser]
        fin_prompt  = prompt_to_model
        print(f"Prompt→diagnoser: {fin_prompt[:80]}...")

        reg1, reward1, out_len1, avg_array, all_rewards_sum, all_rewards_diag = get_regret(
            deployments_diagnoser,
            fin_prompt,
            task,
            arm_select_diagnoser,
            avg_array,
            t,
            all_rewards_sum,
            all_rewards_diag,
            labels,
            dataset
        )

        diag_reward_sums[arm_select_diagnoser] += reward1
        diag_counts     [arm_select_diagnoser] += 1
        diag_avg_array  [arm_select_diagnoser]  = (
            diag_reward_sums[arm_select_diagnoser] /
            diag_counts[arm_select_diagnoser]
        )

        costs += (
            input_cost_per_token[arm_select_diagnoser] * in_lengths[arm_idx_1]
            + cost_per_token[arm_select_diagnoser] * out_len1
        )
        tot_reward += int(reward1)
        all_rewards_diag.append(reward1)
        print(f"Solver Reward: {reward1}  |  Solver Regret: {reg1}")
        summ_regret = reg1 if t == 0 else (regrets[-1] + reg1)
        regrets.append(summ_regret)

        l1 = getattr(
            dyn_vars, 
            f"l_{len(summary_description_array) + arm_idx_1}"
        )
        if t == 0:
            all_rewards_diag[-1] = 1 if reward1 == 0 else reward1
            new_rews  = np.array(all_rewards_diag, dtype=float)
            new_rews  = (new_rews - new_rews.mean()) / new_rews.std() if new_rews.std() != 0 else new_rews
            idx1      = 0
            loss      = l1.train(context[arm_idx_1], new_rews[idx1])
        else:
            new_rews  = np.array(all_rewards_diag, dtype=float)
            new_rews  = (new_rews - new_rews.mean()) / new_rews.std()
            idx1      = len(new_rews) - 1
            loss      = l1.train(context[arm_idx_1], new_rews[idx1])

        # SUBTASK Explanation
        context = []
        values  = []
        models  = ["base", "finetune_med", "finetune_tele", "finetune_med_new", "llama"]

        # Build the “explanation prompt,” appending the chosen MCQ answer
        explanation_prompt = (
            input_reports[t] + "\nAnswer chosen: " + (str(reward1) if isinstance(reward1, int) else "<unknown>" + "Please give a explination for why this answer choice is correct.")
        )
        task    = "explanation"

        documents = (
            summary_description_array
            + diagnosis_description_array
            + explanation_description_array
            + list(input_reports)
            + [summary, explanation_prompt]
        )

        for j in range(len(explanation_description_array)):
            l = getattr(
                dyn_vars,
                f"l_{len(summary_description_array) + len(diagnosis_description_array) + j}"
            )
            cont = get_context(
                documents,
                t,
                2,  # subtask 2
                j,
                len(summary_description_array),
                len(diagnosis_description_array),
                len(input_reports),
                inp_model,
                dataset
            )
            context.append(cont)
            values.append(l.selection(cont, args.style))

        toks = reg_tokenizer(
            explanation_prompt,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(device)

        pred_lengths = []
        for mk in models:
            llm_name = arm_to_llm[mk]
            idx      = orig_model_names.index(llm_name)
            onehot   = torch.zeros(len(orig_model_names), device=device)
            onehot[idx] = 1.0
            onehot   = onehot.unsqueeze(0)
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
            if hasattr(enc, "encode"):
                in_len = len(enc.encode(explanation_prompt))
            else:
                in_len = len(enc(explanation_prompt, truncation=True)["input_ids"])
            in_lengths.append(in_len)

        values_f = [float(v) for v in values]
        print(f"Accuracy UCB (explainer): {values_f}")

        cost = []
        for i, mk in enumerate(models):
            #values[i] = values[i] - 500*cost_per_token[mk] * pred_lengths[i]
            values[i] = (
                values[i]- alpha*cost_per_token[mk]* pred_lengths[i]- alpha*input_cost_per_token[mk] * in_lengths[i]
            )
            cost.append(alpha*cost_per_token[mk]* pred_lengths[i] + alpha*input_cost_per_token[mk] * in_lengths[i])
        print(f"Budget cost: {cost}")

        if args.style == "ts":
            values_np  = [v.cpu().item() if isinstance(v, torch.Tensor) else v for v in values]
            arm_idx_2  = np.random.choice(
                np.where(np.array(values_np) == np.array(values_np).max())[0]
            )
        else:
            arm_idx_2  = int(np.argmax(values))

        plays_no[
            len(summary_description_array)
            + len(diagnosis_description_array)
            + arm_idx_2
        ] += 1

        arm_select_explainer = models[arm_idx_2]
        print(f"Selected explainer: {arm_select_explainer}")

        if arm_select_explainer in ["finetune_med", "finetune_tele", "finetune_med_new"]:
            cat = "finetune"
        else:
            cat = arm_select_explainer

        deployment = deployments_explainer[arm_select_explainer]
        fin_prompt  = explanation_prompt
        print(f"Prompt→explainer: {fin_prompt[:80]}...")

        explanation_text = get_summary(explanation_prompt, arm_select_explainer, "tele")
        out_len2 = len(explanation_text.split())

        gold_explanation = explinations[t]

        reg2, reward2, out_len2_eval, avg_array, all_rewards_sum, all_rewards_expl = get_regret(
            deployments_explainer,
            fin_prompt,
            task,
            arm_select_explainer,
            avg_array,
            t,
            all_rewards_sum,
            all_rewards_expl,
            explinations,
            dataset
        )

        expl_reward_sums[arm_select_explainer] += reward2
        expl_counts     [arm_select_explainer] += 1
        expl_avg_array  [arm_select_explainer]  = (
            expl_reward_sums[arm_select_explainer] /
            expl_counts[arm_select_explainer]
        )

        # Accumulate cost & reward
        costs += (
            input_cost_per_token[arm_select_explainer] * in_lengths[arm_idx_2]
            + cost_per_token[arm_select_explainer] * out_len2
        )
        costs_list.append(costs)
        tot_reward += float(reward2)
        rewards.append(tot_reward)

        print(
            f"Explanation Reward: {reward2:.3f}  |  Explanation Regret: {reg2:.3f} "
            f"|  Generated len: {out_len2}"
        )

        if t == 0:
            expl_regret_cum = reg2
        else:
            expl_regret_cum = regrets[-1] + reg2
        regrets[-1] = expl_regret_cum

        l2 = getattr(
            dyn_vars,
            f"l_{len(summary_description_array) + len(diagnosis_description_array) + arm_idx_2}"
        )
        if t == 0:
            all_rewards_expl[-1] = max(reward2, 1e-3)
            new_rews  = np.array(all_rewards_expl, dtype=float)
            new_rews  = (new_rews - new_rews.mean()) / new_rews.std() if new_rews.std() != 0 else new_rews
            idx2      = 0
            loss      = l2.train(context[arm_idx_2], new_rews[idx2])
        else:
            new_rews  = np.array(all_rewards_expl, dtype=float)
            new_rews  = (new_rews - new_rews.mean()) / new_rews.std()
            normalized_val = (reward2 - new_rews.mean()) / new_rews.std()
            idx2      = len(new_rews) - 1
            loss      = l2.train(context[arm_idx_2], new_rews[idx2])
        
        if (t+1) % 5 == 0:
            print(
                f"After round {t+1}: cum‐regret={regrets[-1]:.3f}, "
                f"cum‐reward={rewards[-1]:.3f}, loss={loss:.3f}"
            )

    all_regrets.append(regrets)
    all_rewards.append(rewards)
    all_costs.append(costs_list)
    all_costs_summarizer.append(costs_list_summarizer)
    all_plays.append(plays_no.copy())
    all_avg_arrays.append(avg_array.copy())
    all_avg_summary_arrays.append(summary_avg_array.copy())
    all_diag_avg_arrays .append(diag_avg_array.copy())
    all_expl_avg_arrays .append(expl_avg_array.copy())

import pandas as pd

regrets_arr            = np.array(all_regrets)
rewards_arr            = np.array(all_rewards)
costs_arr              = np.array(all_costs)
costs_summarizer_arr   = np.array(all_costs_summarizer)
plays_arr              = np.array(all_plays)
avg_df                 = pd.DataFrame(all_avg_arrays)
summary_avg_df         = pd.DataFrame(all_avg_summary_arrays)
diag_avg_df            = pd.DataFrame(all_diag_avg_arrays)
expl_avg_df            = pd.DataFrame(all_expl_avg_arrays)

regrets_mean           = regrets_arr.mean(axis=0)
regrets_std            = regrets_arr.std(axis=0)
rewards_mean           = rewards_arr.mean(axis=0)
rewards_std            = rewards_arr.std(axis=0)
costs_mean             = costs_arr.mean(axis=0)
costs_std              = costs_arr.std(axis=0)
costs_sum_mean         = costs_summarizer_arr.mean(axis=0)
costs_sum_std          = costs_summarizer_arr.std(axis=0)
plays_mean             = plays_arr.mean(axis=0)
plays_std              = plays_arr.std(axis=0)

avg_mean               = avg_df.mean(axis=0).to_dict()
avg_std                = avg_df.std(axis=0).to_dict()
summary_mean           = summary_avg_df.mean(axis=0).to_dict()
summary_std            = summary_avg_df.std(axis=0).to_dict()
diag_mean              = diag_avg_df.mean(axis=0).to_dict()
diag_std               = diag_avg_df.std(axis=0).to_dict()
expl_mean              = expl_avg_df.mean(axis=0).to_dict()
expl_std               = expl_avg_df.std(axis=0).to_dict()

pickle.dump(regrets_mean,   open("tele_results/regrets_mean_tele_3subtasks_seqgpt_2.pkl",   "wb"))
pickle.dump(regrets_std,    open("tele_results/regrets_std_tele_3subtasks_seqgpt_2.pkl",    "wb"))
pickle.dump(rewards_mean,   open("tele_results/rewards_mean_tele_3subtasks_seqgpt_2.pkl",   "wb"))
pickle.dump(rewards_std,    open("tele_results/rewards_std_tele_3subtasks_seqgpt_2.pkl",    "wb"))
pickle.dump(costs_mean,     open("tele_results/costs_mean_tele_3subtasks_seqgpt_2.pkl",     "wb"))
pickle.dump(costs_std,      open("tele_results/costs_std_tele_3subtasks_seqgpt_2.pkl",      "wb"))
pickle.dump(costs_sum_mean, open("tele_results/costs_summarizer_mean_tele_3subtasks_seqgpt_2.pkl", "wb"))
pickle.dump(costs_sum_std,  open("tele_results/costs_summarizer_std_tele_3subtasks_seqgpt_2.pkl",  "wb"))
pickle.dump(plays_mean,     open("tele_results/plays_mean_tele_3subtasks_seqgpt_2.pkl",     "wb"))
pickle.dump(plays_std,      open("tele_results/plays_std_tele_3subtasks_seqgpt_2.pkl",      "wb"))
pickle.dump(avg_mean,       open("tele_results/avg_accuracy_mean_tele_3subtasks_seqgpt_2.pkl","wb"))
pickle.dump(avg_std,        open("tele_results/avg_accuracy_std_tele_3subtasks_seqgpt_2.pkl", "wb"))
pickle.dump(summary_mean,   open("tele_results/sum_avg_accuracy_mean_tele_3subtasks_seqgpt_2.pkl","wb"))
pickle.dump(summary_std,    open("tele_results/sum_avg_accuracy_std_tele_3subtasks_seqgpt_2.pkl", "wb"))
pickle.dump(diag_mean,      open("tele_results/diag_accuracy_mean_tele_3subtasks_seqgpt_2.pkl", "wb"))
pickle.dump(diag_std,       open("tele_results/diag_accuracy_std_tele_3subtasks_seqgpt_2.pkl",  "wb"))
pickle.dump(expl_mean,      open("tele_results/expl_accuracy_mean_tele_3subtasks_seqgpt_2.pkl", "wb"))
pickle.dump(expl_std,       open("tele_results/expl_accuracy_std_tele_3subtasks_seqgpt_2.pkl",  "wb"))


print(f"Final mean regret: {regrets_mean[-1]:.4f}")
print(f"Final mean reward: {rewards_mean[-1]:.4f}")
print(f"Final mean cost: {costs_mean[-1]:.4f}")
print(f"Final mean summarizer cost: {costs_sum_mean[-1]:.4f}")
print(f"Final mean plays: {plays_mean}")
print(f"Final mean avg‐accuracy array: {avg_mean}")
print(f"Final mean summary‐accuracy array: {summary_mean}")
print("All runs complete. Pickles written for 3‐subtask telecom experiments.")
