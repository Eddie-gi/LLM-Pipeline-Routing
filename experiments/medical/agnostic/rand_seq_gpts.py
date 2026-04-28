import numpy as np
from utils.helper import get_optimal_super_arm_reward, get_reward, get_regret, parse_arguments, load_data
from src.prompts.prompt_maker import input_maker
from src.regrets.sum_call import get_summary
import pickle

# Parse command-line arguments
args = parse_arguments()
num_rounds = args.size
no_tasks = args.number_tasks

# Load data and process reports
diagnoses, input_reports = load_data()
input_reports = input_maker("rand", "medical", input_reports)
new_labels = diagnoses

class DynamicVariables:
    pass

dyn_vars = DynamicVariables()
num_rounds = args.size



regrets = []
summ = 0
rew = 0
rewards_list = []
total_reward = 0
summary_description_array = ["Summarize the main points of the medical report of a patient, this summary will be used for research purposes only.",
                             "You will take the role of an diagnosis analyst. Use your knowledge base to summarize inputted medical reports."]

diagnosis_description_array = ["General use LLM which does not specialize in any task specifically.","LLM specializing on medical reports and trained to do medical diagnosis for research purposes.",
                               "LLM specializing for answering multiple choice telecommunications questions.", "LLM specializing on medical reports and trained to do medical diagnosis for research purposes.", "General use small LLM which does not specialize in any task specifically.","General use LLM which does not specialize in any task specifically.", "General use LLM which does not specialize in any task specifically."]
deployments_1 = {"base" : ("gpt-35-turbo","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med" : ("Med","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "finetune_tele" : ("Tele","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,vertebral/basilar stenosis,hypoxia,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"), 
                 "finetune_med_new": ("Med_New","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "small": ("SLM","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "llama": ("llama","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication"),
                 "phi": ("phi","You will take the role of an medical agent whose primary goal is to give diagnosis based on medical reports for experimentation and research purposes only. You need to make inferences based on the provided report to make diagnosis predictions. Output at most 2 diagnoses. If you identify multiple diagnosis output them in a comma separated format like heart failure,colon cancer. Your given diagnoses for the patient must be one of the following: diabetes mellitus,huntington's disease,sepsis,encephalopathy,pulmonary embolism,pulmonary edema,tamponade,mitral stenosis,congestive heart failure,chronic obstructive pulmonary disease,abdominal thoracic aneurysm,neurosarcoidosis,renal failure,svc syndrome,urosepsis,acute myocardial infarction,acute coronary syndrome,carotid stenosis,aortic stenosis,coronary artery disease,osteoarthritis,aortic insufficiency,unstable angina/cath,hyperlipidemia,syncope,complete heart block,intravascular coagulation,septic shock,hepatic failure,pneumonia,pancreatitis,anemia,catheter tip infection,coma,urinary tract infection,wound infection,cerebral artery infarction,hyponatremia,cardiomyopathy,hypoxia,vertebral/basilar stenosis,sick sinus syndrome,pulmonary congestion,aseptic meningitis,neutropenia,cellulitis,cirrhosis of liver,liver failure,pericardial effusion,aortic valve dysfunction,venous thrombosis,respiratory failure,benzodiazepine overdose,vena cava obstruction,valvular heart disease,v-tach,aortic dissection,opiate intoxication")}
deployments_0 = {"base" : ("gpt-35-turbo","You are to summarize an inputted medical report, this summary will be used for research purposes only."), "assistants" : ("Assistant","You are to summarize an inputted medical report, this summary will be used for research purposes only.")}

emb_size = 384
total_len = len(summary_description_array)+len(diagnosis_description_array)
deploy = [deployments_0,deployments_1]
cat = ''
rewards = 0
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
input_reports = list(input_reports)
documents = summary_description_array+diagnosis_description_array+input_reports
from sentence_transformers import SentenceTransformer
inp_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
from transformers import AutoTokenizer, AutoModel

sum_len = len(summary_description_array)
all_rewards_sum = []
all_rewards_diag = []
import random
plays_no = np.ones(total_len)
avg_array = {"gpt-35-turbo":0,"Med":0,"Tele":0,"Med_New":0,"SLM":0,"llama":0,"phi":0}

for t in range(num_rounds):
    for i in range(no_tasks):
        if i==0:
            context = []
            models = ["assistants","base"]
            prompt_to_model = input_reports[t]
            task = 'summary'
            values = []
            arm = random.randint(0,len(models)-1)
            plays_no[arm] += 1 
            arm_select = models[arm]
        else:
            context = []
            values=[]
            models = ["base","finetune_med","finetune_tele","finetune_med_new","small","llama","phi"]
            prompt_to_model = get_summary(input_reports[t],cat)
            prompt_to_model = prompt_to_model.replace('\n','')
            print(prompt_to_model)
            task = 'diagnosis'
            arm = random.randint(0,len(models)-1)
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

        else:
            reward,all_rewards_sum,all_rewards_diag = get_reward(deployment,cat,fin_prompt,task,all_rewards_sum,all_rewards_diag)
            
    if (t+1) % 5 == 0:
        print('{}: {:.3f}, {:.3f}'.format(t+1, summ, rewards))

import pickle
with open("regrets_seqrandom_med_100_5.pkl", "wb") as file:
    pickle.dump(regrets, file)
with open("rewards_seqrandom_med_100_5.pkl", "wb") as file:
    pickle.dump(rewards_list, file)
with open("arm_plays_seqrandom_med_100_5.pkl", "wb") as file:
    pickle.dump(list(plays_no), file)
with open("avg_array_acc_llms_med_100_5.pkl", "wb") as file:
    pickle.dump(avg_array, file)