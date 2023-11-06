import os 
import sys
import pickle
import evaluate

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, '..'))
sys.path.append(project_root)

os.chdir("/work/agoindan/.cache")
os.environ['TRANSFORMERS_CACHE'] = "/work/agoindan/.cache"
os.environ['TORCH_HUB'] = "/work/agoindan/.cache"
os.environ['XDG_CACHE_HOME'] = "/work/agoindan/.cache"

def compute(ground_truth, predictions):
    bertscore = evaluate.load('bertscore')
    results = bertscore.compute(predictions=predictions, references=ground_truth, 
                                model_type="microsoft/deberta-large-mnli", lang="en", 
                                rescale_with_baseline=True)
    results['precision'] = sum(results['precision'])/ len(results['precision'])
    results['recall'] = sum(results['recall'])/ len(results['recall'])
    results['f1'] = sum(results['f1'])/ len(results['f1'])
    return results
    
datasets = ["enrico", "decimer", "flickr30k", "hateful_memes", "memecaps", "newyorkercartoon",
            "slake", "vqarad", "okvqa", "pathvqa", "scienceqa", "ucmerced", "memotion", "visualgen",
            "inat", "face_emotion", "mmimdb", "vqa", "nocaps"]

models = ["instruct_blip", "blip2", "kosmos2", "gill", "openflamingo", "minigpt4"]

cnt = 0
for dset in datasets:
    for model in models:
        fname = f"{model}_{dset}.pkl"
        if os.path.isfile(fname):
            print(fname)
            res = pickle.load(open(fname, "rb"))
            if "metrics" in res:
                metrics = res["metrics"]
            
            if "results" in res:
                metrics = res["results"]

            if "F1-Score" in metrics:
                continue

            if os.path.exists(f"gt_{dset}.pkl"):
                print(dset)
                cnt += 1
                gt_file = pickle.load(open(f"gt_{dset}.pkl", "rb"))
                samples = min(len(res["predictions"]), len(gt_file))
                new_metrics = compute(gt_file[:samples], res["predictions"][:samples])
                res["metrics"] = new_metrics

            pickle.dump(res, open(fname, "wb"))

print(cnt)

