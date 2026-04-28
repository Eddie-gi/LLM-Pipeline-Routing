import pickle
import argparse
import numpy as np
import torch.nn as nn
from src.regrets.optimal_rand_tele import opt_eval
from src.regrets.final_rand_med import final_eval


def load_data(diagnoses_path='data/diagnoses_100.pkl', input_reports_path='data/input_reports_100.pkl'):
    with open(diagnoses_path, 'rb') as f:
        diagnoses = pickle.load(f)
    with open(input_reports_path, 'rb') as f:
        input_reports = pickle.load(f)
    return diagnoses, input_reports

def parse_arguments():
    parser = argparse.ArgumentParser(description='NeuralUCB Experiment')
    parser.add_argument('--size', default=100, type=int, help='number of rounds')
    parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
    parser.add_argument('--lamdba', type=float, default=1, metavar='l', help='lambda for regularization')
    parser.add_argument('--hidden', type=int, default=50, help='network hidden size')
    parser.add_argument('--style', default='ucb', metavar='ts|ucb', help='TS or UCB')
    parser.add_argument('--number_tasks', default=2, type=int, help='number of tasks')
    return parser.parse_args()

#wrappers
def get_optimal_super_arm_reward(deployments,prompt,task):
    #call Azure API to get optimal reward by trying out all combinations
    return opt_eval(deployments, prompt,task)

def get_reward(deployment,cat,prompt,task,all_rewards_sum,all_rewards_diag):
    
    #call Azure API to get final reward
    return final_eval(deployment, cat, prompt,task,all_rewards_sum,all_rewards_diag)
    

def get_regret(deployments,prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset):
    return opt_eval(deployments, prompt,task,selected,avg_array,t,all_rewards_sum,all_rewards_diag,labels,dataset)