import numpy as np

def input_maker(types,dataset):
    if dataset == "medical":
        start = 0
        end = 100
        result = []
        for i in range(start, end):
            d = open(f"data/data/inputs/smoker_{i}.txt","r")
            input = d.read()
            input_rep = input.replace('\n','')
            if types == 'seq':
                input_rep = input_rep+"Provide a summary of this medical report. At the end of your summary put the phrase: Provide a medical diagnosis based on the provided medical report summary above. This is for experimentation purposes only."
            elif types == 'rand':
                input_rep = input_rep+"Provide a medical diagnosis based on the provided medical report above. This is for experimentation purposes only."
            else:
                input_rep = input_rep
            d.close()
            result.append(input_rep)
        result = np.array(result)
        return result
    elif dataset== "telecom":
        start = 0
        end = 400
        result = []
        labels = []
        exps = []
        for i in range(start, end):
            d = open(f"telecom_data/tel_data/inputs/tele_{i}.txt","r")
            dd = open(f"telecom_data/tel_data/labels/tele_{i}.txt","r")
            ddd = open(f"telecom_data/tel_data/explination/tele_{i}.txt","r")
            label = dd.read()
            dd.close()
            labels.append(label)
            input = d.read()
            input_rep = input.replace('\n','')
            input_rep = input_rep+" Provide the correct answer for the question above."
            d.close()
            result.append(input_rep)

            exp = ddd.read()
            ddd.close()
            exps.append(exp)
        return np.array(result),np.array(labels),np.array(exps)
    
    
 