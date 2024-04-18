import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.stats import ttest_rel

def scale_scores(vals, mode="bert"):
    """
    scaling the raw metrics on a scale of 0 to 1 
    0 being the performance of the model achieving the lowest score 
    1 being the human/orcale performance
    """
    vals = np.array(vals)
    if mode == "bert":
        max_val = 1
    elif mode == "bart":
        max_val = 0
    
    min_val = min(vals)
    vals = vals - min_val
    vals = vals / (max_val - min_val)
    return vals


def mean_and_var(dsets, model_analysis):
    """
    variance of scores on the datasets 
    sorts the datasets by the decreasing variance
    """
    models = ["blip2", "instruct_blip", "minigpt4", "fuyu", "emu", "openflamingo", 
              "kosmos2", "mplugowl", "llama_adapter", "gemini", "gptv"]
    
    variances = []
    for dset in dsets:
        scores = []
        for model in models:
            scores.append(model_analysis[model][dset])
        
        scores = np.array(scores)
        var = np.std(scores)
        variances.append((var, dset))

    variances.sort()
    print(variances[::-1])


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
            idx = df[df["Models"] == model].index
            model_scores.append(float(df[dset][idx]))
            evaluated_models.append(model)
                
        scores = scale_scores(model_scores, mode=mode)

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
        for dset in dsets:
            avg += model_analysis[model][dset]

        avg /= len(dsets)
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


