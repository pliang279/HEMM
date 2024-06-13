import pickle 
import json 
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import random

def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for battle in battles:
        model_a, model_b, winner = battle
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "1":
            sa = 1
        elif winner == "2":
            sa = 0
        elif winner == "3":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating

def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        random.shuffle(battles)
        rows.append(func_compute_elo(battles))
    df = pd.DataFrame(rows)
    print(df.head())
    print(df.median())  
    return df[df.median().sort_values(ascending=False).index]

models = ["blip2", "instruct_blip", "fuyu", "gptv", "gemini", "mplugowl",
          "emu", "openflamingo", "kosmos2", "minigpt4", "llama_adapter"]

wins = np.zeros((len(models), len(models)))
ties = np.zeros((len(models), len(models)))

md = pickle.load(open(f"matchup_data.pkl", "rb"))
total_matches = np.zeros((len(models), len(models)))
battles = []
for i in range(5):
    res = json.load(open(f"results_set{i+1}.json"))
    for idx, out in res.items():
        mod1 = md[int(idx)]["model1"]
        mod2 = md[int(idx)]["model2"]
        idx1 = models.index(mod1)
        idx2 = models.index(mod2)

        total_matches[idx1][idx2] += 1
        total_matches[idx2][idx1] += 1

        if int(out) == 1:
            wins[idx1][idx2] += 1
            battles.append([mod1, mod2, "1"])

        elif int(out) == 2:
            wins[idx2][idx1] += 1
            battles.append([mod1, mod2, "2"])

        else:
            ties[idx1][idx2] += 1
            ties[idx2][idx1] += 1
            battles.append([mod1, mod2, "3"])

print(total_matches)
pair_wise_win_perc = wins / (total_matches - ties)
np.fill_diagonal(pair_wise_win_perc, 0)
np.set_printoptions(precision=2)

print(pair_wise_win_perc)
avg_win_rate = pair_wise_win_perc.mean(axis=1)
indices = avg_win_rate.argsort()[::-1]
# print(indices)
print(avg_win_rate[indices])
print(np.array(models)[indices])
# for i in range(len(models)):
#     print(models[i], win_perc[i])

elo_ratings = get_bootstrap_result(battles, compute_elo, 1000)
# print(elo_ratings)

np.random.seed(42)
# bootstrap_elo_lu = get_bootstrap_result(battles, compute_elo, BOOTSTRAP_ROUNDS)
bootstrap_lu_median = elo_ratings.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
print(bootstrap_lu_median)
