import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.stats import ttest_rel
from utils import *
from categories import *

df = pd.read_csv("./bartscore_100_samples.csv")
df = df.drop(columns=["magicbrush"])

df = df.fillna(0)
datasets = list(df.columns[2:])
mode = "bart"

all_results, model_analysis = results_on_dsets(df, datasets, mode=mode)

dset_scores = []
for dset, score in all_results.items():
    dset_scores.append((score, dset))

dset_scores.sort()

print("Model Average Scores")
model_avg_scores = {}
for model, scores in model_analysis.items():
    avg_sc = 0
    for dset, score in scores.items():
        avg_sc += score
    avg_sc /= len(scores)
    model_avg_scores[model] = avg_sc
    print(f"{model}: {avg_sc}")

print("Category scores")
for dimension in model_categories:
    print(dimension)
    for group in model_categories[dimension]:
        models = model_categories[dimension][group]
        avg_sc = []
        for model in models:
            if model not in model_avg_scores:
                continue
        
            avg_sc.append(model_avg_scores[model])
        
        if len(avg_sc) > 0:
            avg_sc = sum(avg_sc) / len(avg_sc)
            print(f"Average score for {group} is {avg_sc:.4f}")


models1 = model_categories["diversity"]["diverse"]
models2 = model_categories["diversity"]["non_diverse"]

exkn = []
noexkn = []
for dset in datasets:
    scores1 = []
    scores2 = []
    for model in models1:
        if model not in model_analysis:
            continue
        if dset not in model_analysis[model]:
            continue
        scores1.append(model_analysis[model][dset])
    for model in models2:
        if model not in model_analysis:
            continue
        if dset not in model_analysis[model]:
            continue
        scores2.append(model_analysis[model][dset])
    
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    exkn.append(scores1.mean())
    noexkn.append(scores2.mean())


exkn = np.array(exkn)
noexkn = np.array(noexkn)
print(exkn.min(), exkn.max())
print(noexkn.min(), noexkn.max())
print(exkn.mean(), noexkn.mean())
t_stat, p_val = ttest_rel(exkn, noexkn)
std1 = exkn.std()
std2 = noexkn.std()
print("Standard Deviation for Diverse Models",  std1)
print("Standard Deviation for Non Diverse Models",  std2)
print("P-value", p_val)


dset1 = dataset_categories["information_flow"]["querying"]
dset2 = dataset_categories["information_flow"]["fusion"]
exkn = []
noexkn = []

for model in model_analysis:
    scores1 = []
    scores2 = []
    for dset in dset1:
        if dset not in model_analysis[model]:
            continue
        scores1.append(model_analysis[model][dset])

    for dset in dset2:
        if dset not in model_analysis[model]:
            continue
        scores2.append(model_analysis[model][dset])
    
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    exkn.append(scores1.mean())
    noexkn.append(scores2.mean())
    # print(model, scores1.mean(), scores2.mean())

exkn = np.array(exkn)
noexkn = np.array(noexkn)
t_stat, p_val = ttest_rel(exkn, noexkn)
std1 = exkn.std()
std2 = noexkn.std()
print("Standard Deviation for Querying Datasets",  std1)
print("Standard Deviation for Fusion Datasets",  std2)
print("P-value", p_val)

# plot for performance vs pre-training data size
plot_clusters(None, model_avg_scores)

plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(35, 40))
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(params)

bar_values = []
legend_labels = ["Multimedia", "Affect", "Science", "Health", "HCI"]
color_mapping = {
    "Multimedia": "#0a9396",  
    "Affect": "#ee9b00",  
    "Science": "#bdb2ff",  
    "Health": "#7f7f7f",  
    "HCI": "#e34a33"  
}

legend_colors = [color_mapping[label] for label in legend_labels]

for i, legend in enumerate(legend_labels):
    labels = dataset_categories["use_case"][legend.lower()]
    bar_values = []
    for dset in labels:
        scores = []
        for model in model_analysis:
            if dset not in model_analysis[model]:
                continue
            scores.append(model_analysis[model][dset])
        avg_sc = sum(scores) / len(scores)
        bar_values.append(avg_sc)
    labels = [dset_to_name[label] for label in labels]
    plt.barh(labels, bar_values, color=legend_colors[i], label=legend)

plt.ylabel('Datasets', fontsize=50)
plt.xlabel('Average Performance of Models', fontsize=50)
plt.title('Average Scores for Use Cases', fontsize=50)
plt.yticks(rotation=15, fontsize=50, ha="right")
plt.xticks(fontsize=50)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels), fontsize=50)

plt.xlim(0, 0.6)
plt.tight_layout()
plt.savefig("use_case.png")
plt.show()

