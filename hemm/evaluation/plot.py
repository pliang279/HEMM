import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.stats import ttest_rel
from utils import *
from categories import *

df = pd.read_csv("/Users/akshay/hemm_res/small_res/eval_100/bartscore_100_samples.csv")
df = df.drop(columns=["magicbrush"])

df = df.fillna(0)
print(df.head())
datasets = list(df.columns[2:])
mode = "bart"
all_results, model_analysis = results_on_dsets(df, datasets, mode=mode)

dset_scores = []
for dset, score in all_results.items():
    dset_scores.append((score, dset))

dset_scores.sort()

for i in range(len(dset_scores)):
    print(dset_scores[i][1], dset_scores[i][0])

print("*"*100)

model_avg_scores = {}
for model, scores in model_analysis.items():
    avg_sc = 0
    for dset, score in scores.items():
        avg_sc += score
    avg_sc /= len(scores)
    model_avg_scores[model] = avg_sc
    print(f"{model}: {avg_sc}")

for dset, val in model_analysis["gptv"].items():
    print("#"*100)

    print(dset)
    print("GPTV", val)
    print("Gemini", model_analysis["gemini"][dset])
    print("BLIP2", model_analysis["blip2"][dset])
    print("Instruct BLIP", model_analysis["instruct_blip"][dset])
    print("Fuyu", model_analysis["fuyu"][dset])

for dimension in model_categories:
    print(dimension)
    for group in model_categories[dimension]:
        models = model_categories[dimension][group]
        avg_sc = 0
        for model in models:
            avg_sc += model_avg_scores[model]
        avg_sc /= len(models)
        print(f"Average score for {group} is {avg_sc:.4f}")

dsets = dataset_categories["use_case"]["science"]

for dset in dsets:
    print(dset)
    for model in model_analysis:
        print(f"{model:}", model_analysis[model][dset])

    print("*"*100)

# plot for performance vs pre-training data size
# num_params_performance = []
# model_names = []
# for model in model_avg_scores:
#     model_names.append(model)
#     num_params_performance.append((tota_number_of_params[model], model_avg_scores[model]))


# plt.style.use('seaborn-darkgrid')
# plt.figure(figsize=(12, 6))
# plt.scatter([x[0] for x in num_params_performance], [x[1] for x in num_params_performance])
# plt.xlabel('Number of Parameters')
# plt.ylabel('Average Scores')
# plt.title('Performance vs Number of Parameters')
# plt.xticks(rotation=45)
# for i, txt in enumerate(model_names):
#     plt.annotate(txt, (num_params_performance[i][1], num_params_performance[i][0]))

# plt.show()


# for dimension in dataset_categories:
#     print(dimension)
#     for group in dataset_categories[dimension]:
#         dsets = dataset_categories[dimension][group]
#         avg_score = avg(model_analysis, dsets)
#         print(f"Average score for {group} is {avg_score.mean():.4f}")
        
