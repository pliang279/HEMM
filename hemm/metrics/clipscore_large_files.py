import os
os.chdir("/work/agoindan/.cache")
os.environ['TRANSFORMERS_CACHE'] = "/work/agoindan/.cache"
os.environ['TORCH_HUB'] = "/work/agoindan/.cache"
os.environ['XDG_CACHE_HOME'] = "/work/agoindan/.cache"

import sys
sys.path.append("/home/agoindan/HEMM")

from tqdm import tqdm
from glob import glob
import pickle 
 
from hemm.utils.base_utils import load_dataset_evaluator
from hemm.metrics.clipscore_metric import CLIPMetric
from hemm.metrics.accuracy_metric import *


datasets = load_dataset_evaluator(kaggle_api_path="/home/agoindan/")
metric = CLIPMetric("cuda")
models = ["blip2", "instruct_blip", "openflamingo", "llama_adapter", "mplugowl", "fuyu", "emu", "minigpt4", "kosmos2"]

chunks = glob("gqa_*.pkl")
for mod in tqdm(models, total=len(models)):
    fn = f"{mod}_gqa.pkl"
    if not os.path.exists(fn):
        continue
    res = pickle.load(open(fn, "rb"))
    if metric.name in res["results"]:
        continue
    scores = []
    for i in range(len(chunks)):
        data = pickle.load(open(f"gqa_{i+1}.pkl", "rb"))
        start_idx = 15000 * i
        end_idx = 15000 * (i+1)
        chunk_score = metric.compute(data["images"], res["predictions"][start_idx:end_idx], data["gts"])
        scores.append(chunk_score * len(data["images"]))

    model_score = sum(scores) / len(res["predictions"])
    res["results"][metric.name] = model_score

    pickle.dump(res, open(fn, "wb"))
