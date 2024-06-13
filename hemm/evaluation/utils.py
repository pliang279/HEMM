import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.stats import ttest_rel
import math
from categories import *
import pickle

datasets_to_skip_for_gptv = ["hatefulmemes", "plip", "memotion"]
dset_max_scores = pickle.load(open("dset_max_scores.pkl", "rb"))

def scale_scores(vals, dset, mode="bert"):
    """
    scaling the raw metrics on a scale of 0 to 1 
    0 being the performance of the model achieving the lowest score 
    1 being the human/orcale performance
    """
    vals = np.array(vals)
    if mode == "bert":
        max_val = 1
    elif mode == "bart":
        max_val = dset_max_scores[dset]["Bart Score"]
    
    min_val = min(vals)
    vals = vals - min_val
    vals = vals / (max_val - min_val)
    return vals


def results_on_dsets(df, datasets, mode="bert"):
    """
    reading the raw scores from the pandas df and returning the average score for each dataset 
    and the normalized/scaled score for all models on all datasets
    returns:
        dset_scores: average score of all the models on each dataset
        model_analysis: scaled score of all models on all datasets
    """
    model_analysis = {}
    dset_scores = {}
    models = list(df["Models"])

    for dset in datasets:
        model_scores = []
        evaluated_models = []
        for model in models:
            if model == "gptv" and dset in datasets_to_skip_for_gptv:
                continue
            idx = df[df["Models"] == model].index
            model_scores.append(float(df[dset][idx]))
            evaluated_models.append(model)
                
        scores = scale_scores(model_scores, dset, mode=mode)

        for i, model in enumerate(evaluated_models):
            if model not in model_analysis:
                model_analysis[model] = {}
            model_analysis[model][dset] = scores[i]

        avg_model_score = scores.mean()
        dset_scores[dset] = avg_model_score

    return dset_scores, model_analysis

def avg(model_analysis, dsets):
    """
    parameters:
        model_analysis: normalized score of every model on every dataset
        dsets: list of dataset names of interest.
    return:
        avg_scs: np.array with the mean score of each model on all the datasets.
    """
    avg_scs = []
    for model in model_analysis:
        avg = 0
        total = 0
        for dset in dsets:
            if dset not in model_analysis[model]:
                continue
            avg += model_analysis[model][dset]
            total += 1

        avg /= total
        avg_scs.append(avg)
    avg_scs = np.array(avg_scs)
    return avg_scs

def compare_avg(list1, list2, model_avg_scores):
    """
    Comparing the average performance of the models on datasets in two groups: list1 and list2
    arguments:
        list1: average score of models on datasets in list1
        list2: average score of models on datasets in list2
        model_avg_scores: average score of models on the datasets
    """
    v1 = []
    v2 = []
    for dt in list1:
        v1.append(model_avg_scores[dt])

    for dt in list2:
        v2.append(model_avg_scores[dt])

    v1 = np.array(v1)
    v2 = np.array(v2)
    print("Avg", v1.mean(), v2.mean())

def compare_avg3(list1, list2, list3, model_avg_scores):
    v1 = []
    v2 = []
    v3 = []
    for dt in list1:
        v1.append(model_avg_scores[dt])

    for dt in list2:
        v2.append(model_avg_scores[dt])
    
    for dt in list3:
        v3.append(model_avg_scores[dt])

    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    print("Avg", v1.mean(), v2.mean(), v3.mean())

def plot_clusters(dimension, model_avg_scores):
    color_mapping = {
            "diverse": "#FFD700",
            "non_diverse": "#FFA07A",
            "instruction_tuning": "#20B2AA",
            "supervised_fine_tuning": "#9370DB",
            "interleaved": "#32CD32",
            "separate": "#4169E1",
            "end_to_end": "#FF1493",
            "modular_fine_tune": "#FF6347",
            "small": "#FF4500",
            "medium": "#00CED1",
            "large": "#FFD700"
    }
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 6))
    if dimension is None:
        num_params_performance = []
        model_names = []
        for model in model_avg_scores:
            model_names.append(model_to_name[model])
            num_params_performance.append((math.log(total_number_of_params[model]), model_avg_scores[model]))

        for i, txt in enumerate(model_names):
            if "MiniGPT" in model_names[i]:
                plt.annotate(txt, (num_params_performance[i][0] + 0.1, num_params_performance[i][1]), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
            elif "LLaMA" in model_names[i]:
                plt.annotate(txt, (num_params_performance[i][0] - 0.1, num_params_performance[i][1] - 0.03), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
            elif "gemini" in model_names[i].lower() or "gpt" in model_names[i].lower():
                plt.annotate(txt, (num_params_performance[i][0] - 0.1, num_params_performance[i][1] - 0.02), 
                             textcoords="offset points", xytext=(0,10), ha='right', fontsize=12)
            elif "kosmos" in model_names[i].lower():
                plt.annotate(txt, (num_params_performance[i][0], num_params_performance[i][1]), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
            else:
                plt.annotate(txt, (num_params_performance[i][0] - 0.1, num_params_performance[i][1]), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)


        plt.scatter([x[0] for x in num_params_performance], [x[1] for x in num_params_performance], c="#20B2AA", s=50)
        plt.xlabel('Log (total number of parameters (in billions))', fontsize=15)
        plt.ylabel('Average Scores', fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Performance vs Number of Parameters', fontsize=15)

        # plt.legend()
        plt.show()
        return

    for category in model_categories[dimension]:
        num_params_performance = []
        models = model_categories[dimension][category]
        model_names = []
        for model in models:
            model_names.append(model)
            num_params_performance.append((math.log(total_number_of_params[model]), model_avg_scores[model]))

        for i, txt in enumerate(model_names):
            plt.annotate(txt, (num_params_performance[i][0] - 0.1, num_params_performance[i][1] - 0.01))

        plt.scatter([x[0] for x in num_params_performance], [x[1] for x in num_params_performance], c=color_mapping[category], label=category)

    plt.xlabel('Log (total number of parameters (in billions))')
    plt.ylabel('Average Scores')
    plt.title('Performance vs Number of Parameters')

    plt.legend()
    plt.show()

