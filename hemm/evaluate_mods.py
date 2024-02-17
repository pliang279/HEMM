import os 
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, '..'))
sys.path.append(project_root)

os.chdir("/work/agoindan/.cache")
os.environ['TRANSFORMERS_CACHE'] = "/work/agoindan/.cache"
os.environ['TORCH_HUB'] = "/work/agoindan/.cache"
os.environ['XDG_CACHE_HOME'] = "/work/agoindan/.cache"

import pickle 
from PIL import Image
from tqdm import tqdm
import re
from hemm.utils.base_utils import load_dataset_evaluator
from hemm.metrics.accuracy_metric import * 
from hemm.metrics.bertscore_metric import * 

def answer_extractor(text, dataset_key):
    if dataset_key == 'hateful_memes' or dataset_key =='newyorkercartoon' or dataset_key =='irfl':
        text = text[:3]
        text = text.lower().strip()
        text = ''.join(filter(str.isalpha, text.lower()))
        return text
    elif dataset_key == 'memotion' or dataset_key == 'face_emotion'  or dataset_key == 'scienceqa' or dataset_key == 'vcr':
        match = re.search(r"\b\d\b", text)
        if match:
            first_number = int(match.group())
            return first_number
        else:
            return None
        
def help(preds, key):
    all = []
    for pred in preds:
        ans = answer_extractor(pred, key)
        if "yes" in ans:
            all.append(1)
        else:
            all.append(0)
    
    return all

datasets = load_dataset_evaluator(kaggle_api_path="/home/agoindan/")

models = ["blip2", "instruct_blip", "minigpt4", "openflamingo", "llama_adapter", "mplugowl", "fuyu", "emu", "minigpt4"]

for dset in tqdm(datasets.keys(), total=len(datasets)):
    for mod in models:
        if "emu" not in mod:
            continue
        fn = f"{mod}_{dset}.pkl"
        if not os.path.exists(fn):
            continue

        data = pickle.load(open(fn, "rb"))
        if "results" in data:
            if len(data["results"]) != 0:
                continue
        
        if "metrics" in data:
            if len(data["metrics"]) != 0:
                continue
        
        # print(len(data["predictions"]), len(data["gts"]))
        print(mod, dset)

        results = {}
        for metric in datasets[dset].metrics:
            if isinstance(metric, AccuracyMetric) or isinstance(metric, PrecisionMetric) or isinstance(metric, RecallMetric) or isinstance(metric, F1ScoreMetric):
                preds = help(data["predictions"], dset)
                if "gt" in data:
                    results[metric.name] = metric.compute(preds, data["gt"])
                elif "gts" in data:
                    results[metric.name] = metric.compute(preds, data["gts"])
                continue

            if "gt" in data:
                results[metric.name] = metric.compute(data["predictions"], data["gt"])
            elif "gts" in data:
                results[metric.name] = metric.compute(data["predictions"], data["gts"])

        if len(results) != 0:
            data["results"] = results
            print("Done", fn)
            pickle.dump(data, open(fn, "wb"))

    