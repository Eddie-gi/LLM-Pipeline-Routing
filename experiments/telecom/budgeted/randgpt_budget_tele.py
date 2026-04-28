import numpy as np
import pickle
import argparse
from src.prompts.prompt_maker import input_maker
import random

from utils.helper import opt_eval, get_summary

input_reports, labels, explanations = input_maker("seq", "telecom")
dataset = "telecom"

deployments_summarizer = {
    "base"            : ("gpt-35-turbo", "You are to summarize a telecom question and its options."),
    "assistants"      : ("Assistant",     "You are to summarize a telecom question and its options."),
    "finetune_med"    : ("Med",           "You are to summarize a telecom question and its options."),
    "finetune_tele"   : ("Tele",          "You are to summarize a telecom question and its options."),
    "finetune_med_new": ("Med_New",       "You are to summarize a telecom question and its options."),
    "llama"           : ("llama",         "You are to summarize a telecom question and its options.")
}

deployments_diagnoser = {
    "base"            : ("gpt-35-turbo", "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}' where i ∈ {1,2,3,4}."),
    "finetune_med"    : ("Med",          "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'."),
    "finetune_tele"   : ("Tele",         "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'."),
    "finetune_med_new": ("Med_New",      "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'."),
    "llama"           : ("llama",        "You are to answer multiple choice questions related to telecommunications. Output strictly 'option {i}'.")
}

deployments_explainer = {
    "base"            : ("gpt-35-turbo", "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "finetune_med"    : ("Med",          "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "finetune_tele"   : ("Tele",         "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "finetune_med_new": ("Med_New",      "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale."),
    "llama"           : ("llama",        "You are to explain why the MCQ answer for this telecom question is correct. Provide a detailed rationale.")
}

models_summarizer = list(deployments_summarizer.keys())
models_diagnoser  = list(deployments_diagnoser.keys())
models_explainer  = list(deployments_explainer.keys())

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

parser = argparse.ArgumentParser(description="Random baseline (telecom, 3 subtasks)")
parser.add_argument("--size",         default=100, type=int, help="number of rounds")
parser.add_argument("--no_runs",      default=2,   type=int, help="how many independent runs")
args, _ = parser.parse_known_args()

num_rounds = args.size
no_runs    = args.no_runs

all_regrets  = []  # per run: list of cumulative regrets per round
all_rewards  = []  # per run: total reward (diag + expl)
all_costs    = []  # per run: cost evolution per round
all_plays    = []  # per run: play counts per arm across 3 subtasks (concatenated)
all_avg_acc  = []  # per run: average accuracy per model (diagnosis + explanation)

for run in range(no_runs):
    print(f"\n===== Starting random-run {run+1}/{no_runs} =====")
    regrets = []
    rewards = []
    costs_list = []
    plays_no = np.ones(len(models_summarizer) + len(models_diagnoser) + len(models_explainer))
    avg_acc             = {
        "gpt-35-turbo": 0.0,
        "Med"         : 0.0,
        "Tele"        : 0.0,
        "Med_New"     : 0.0,
        "llama"       : 0.0
    }

    # Track per-arm counts and rewards (for accuracy averaging)
    counts_diag = {arm: 0 for arm in models_diagnoser}
    sums_diag = {arm: 0.0 for arm in models_diagnoser}
    counts_expl = {arm: 0 for arm in models_explainer}
    sums_expl = {arm: 0.0 for arm in models_explainer}

    all_rewards_sum = []  # for normalization in seqgpt, but here unused for summarizer (stub)
    all_rewards_diag = []
    all_rewards_expl = []

    total_cost = 0.0
    total_reward = 0.0
    cum_regret = 0.0

    for t in range(num_rounds):
        print(f"Round {t+1}")
        # ────── Subtask 0: Summarization (random) ──────────────────────────────
        prompt = input_reports[t].replace("\n", " ")
        # Randomly select summarizer arm
        arm_idx_sum = random.randrange(len(models_summarizer))
        arm_sum = models_summarizer[arm_idx_sum]
        plays_no[arm_idx_sum] += 1

        # Generate summary via get_summary
        summary = get_summary(input_reports[t], arm_sum, "tele")
        # Cost for summarizer
        in_len_sum = len(summary.split())  # approximation: word count as token proxy
        total_cost += input_cost_per_token[arm_sum] * in_len_sum + cost_per_token[arm_sum] * in_len_sum
        costs_list.append(total_cost)
        # Summarizer reward stub = 0, no training

        # Prepare prompt for diagnosis including summary
        diag_prompt = prompt + "\n\nSummary:\n" + summary + "\nPlease give the correct option in the format: option [number]."

        # ────── Subtask 1: Diagnosis (random) ──────────────────────────────────
        arm_idx_diag = random.randrange(len(models_diagnoser))
        arm_diag = models_diagnoser[arm_idx_diag]
        # Play index offset for diagnosis arms
        plays_no[len(models_summarizer) + arm_idx_diag] += 1

        # Get regret and reward for diagnosis
        reg1, reward1, out_len1, avg_acc, _, _ = opt_eval(
            deployments_diagnoser,
            diag_prompt,
            "diagnosis",
            arm_diag,
            avg_acc,         # avg_array reused to accumulate
            t,
            all_rewards_sum,  # sum across all subtasks
            all_rewards_diag, # tracking diag rewards
            labels,
            dataset
        )
        # Update cumulative regret
        cum_regret += reg1
        regrets.append(cum_regret)
        total_reward += reward1

        # Update diagnosis rewards and counts
        all_rewards_diag.append(reward1)
        counts_diag[arm_diag] += 1
        sums_diag[arm_diag] += reward1

        # Cost for diagnosis
        in_len_diag = len(diag_prompt.split())  # approximation: word count as token proxy
        total_cost += input_cost_per_token[arm_diag] * in_len_diag + cost_per_token[arm_diag] * out_len1
        costs_list[-1] = total_cost  # update cost at this round

        # ────── Subtask 2: Explanation (random) ────────────────────────────────
        # Build explanation prompt
        explanation_prompt = (
            input_reports[t] + "\nAnswer chosen: " + str(reward1) + " Please give an explanation for why this answer choice is correct."
        )
        arm_idx_expl = random.randrange(len(models_explainer))
        arm_expl = models_explainer[arm_idx_expl]
        # Play index offset for explainer arms
        plays_no[len(models_summarizer) + len(models_diagnoser) + arm_idx_expl] += 1

        # Generate explanation text (unused for reward computation)
        explanation_text = get_summary(explanation_prompt, arm_expl, "tele")
        out_len2 = len(explanation_text.split())

        # Get gold explanation
        gold_expl = explanations[t]
        # Get regret and reward for explanation
        reg2, reward2, out_len2_eval, avg_acc, _, _ = opt_eval(
            deployments_explainer,
            explanation_prompt,
            "explanation",
            arm_expl,
            avg_acc,
            t,
            all_rewards_sum,
            all_rewards_expl,
            explanations,
            dataset
        )
        # Update cumulative regret
        cum_regret += reg2
        regrets[-1] = cum_regret  # reflect both subtasks
        total_reward += reward2

        # Update explanation rewards and counts
        rewards.append(total_reward)
        all_rewards_expl.append(reward2)
        counts_expl[arm_expl] += 1
        sums_expl[arm_expl] += reward2

        # Cost for explanation
        in_len_expl = len(explanation_prompt.split())
        total_cost += input_cost_per_token[arm_expl] * in_len_expl + cost_per_token[arm_expl] * out_len2
        costs_list[-1] = total_cost

        # End of round
        if (t + 1) % 50 == 0:
            print(f"Round {t+1}: cum-regret={cum_regret:.3f}, total-cost={total_cost:.3f}")

    all_regrets.append(np.array(regrets))
    all_rewards.append(np.array(rewards))
    all_costs.append(np.array(costs_list))
    all_plays.append(plays_no.copy())
    all_avg_acc.append(avg_acc.copy())

# ─── FINAL AGGREGATION AND PICKLE SAVING ─────────────────────────────────────────
import pandas as pd
regrets_arr = np.stack(all_regrets, axis=0)  # shape (no_runs, num_rounds)
rewards_arr = np.stack(all_rewards, axis=0)
costs_arr   = np.stack(all_costs, axis=0)
plays_arr   = np.stack(all_plays, axis=0)      # shape (no_runs, total_arms)
avg_df      = pd.DataFrame(all_avg_acc)

# Compute means and stds
regrets_mean = regrets_arr.mean(axis=0)
regrets_std  = regrets_arr.std(axis=0)
rewards_mean = rewards_arr.mean(axis=0)
rewards_std  = rewards_arr.std(axis=0)
costs_mean    = costs_arr.mean(axis=0)
costs_std     = costs_arr.std(axis=0)
plays_mean    = plays_arr.mean(axis=0)
plays_std     = plays_arr.std(axis=0)
avg_mean      = avg_df.mean(axis=0).to_dict()
avg_std       = avg_df.std(axis=0).to_dict()

# Save pickles for comparison
pickle.dump(regrets_mean, open("tele_results/regrets_mean_tele_rand_3subtasks_1.pkl", "wb"))
pickle.dump(regrets_std,  open("tele_results/regrets_std_tele_rand_3subtasks_1.pkl",  "wb"))
pickle.dump(rewards_mean, open("tele_results/rewards_mean_tele_randgpt_3subtasks_1.pkl","wb"))
pickle.dump(rewards_std,  open("tele_results/rewards_std_tele_randgpt_3subtasks_1.pkl","wb"))
pickle.dump(costs_mean,   open("tele_results/costs_mean_tele_rand_3subtasks_1.pkl",   "wb"))
pickle.dump(costs_std,    open("tele_results/costs_std_tele_rand_3subtasks_1.pkl",    "wb"))
pickle.dump(plays_mean,   open("tele_results/plays_mean_tele_rand_3subtasks_1.pkl",   "wb"))
pickle.dump(plays_std,    open("tele_results/plays_std_tele_rand_3subtasks_1.pkl",    "wb"))
pickle.dump(avg_mean,     open("tele_results/avg_accuracy_mean_tele_rand_3subtasks_1.pkl", "wb"))
pickle.dump(avg_std,      open("tele_results/avg_accuracy_std_tele_rand_3subtasks_1.pkl",  "wb"))

print(f"Final cum-regret (mean): {regrets_mean[-1]:.4f}")
print("Random baseline runs complete. Pickles written for 3-subtask telecom.")
