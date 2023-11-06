import os 
import sys
import pickle as pkl
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, '..'))
sys.path.append(project_root)

os.chdir("/work/agoindan/.cache")
os.environ['TRANSFORMERS_CACHE'] = "/work/agoindan/.cache"
os.environ['TORCH_HUB'] = "/work/agoindan/.cache"
os.environ['XDG_CACHE_HOME'] = "/work/agoindan/.cache"

from hemm.utils.base_utils import load_model, load_dataset_evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple command-line argument example")

    parser.add_argument("--model_key", help="Model key to evaluate all the datasets", type=str)
    parser.add_argument("--batch_size", help="batch size for the model", type=int)
    
    args = parser.parse_args()

    model = load_model(args.model_key)
    model.load_weights()

    print("Model Loaded")

    dataset_evaluator = load_dataset_evaluator(kaggle_api_path="/home/agoindan/")
    print("Datasets Loaded")

    dataset_evaluator = dict(sorted(dataset_evaluator.items(), key=lambda item: len(item[1])))

    # evaluated = ["nocaps", "magic_brush", "decimer", "enrico", "flickr30k", "hateful_memes", "memecaps",
    #             "newyorkercartoon", "slake", "ucmerced", "vqarad"]

    # evaluated = ["enrico", "decimer", "newyorkercartoon", "magic_brush"]

    evaluated = ["enrico", "decimer", "flickr30k", "hateful_memes", "magic_brush", "memecaps", "newyorkercartoon",
                 "slake", "vqarad", "okvqa", "pathvqa", "scienceqa", "ucmerced", "memotion"]

    for name, dataset in dataset_evaluator.items():
        print(f"Evaluating for {name} dataset")
        if name in evaluated:
            continue
        # print(name)
        predictions, results, gt = dataset.evaluate_dataset_batched(model=model, batch_size=args.batch_size)
        # predictions, results = dataset.evaluate_dataset(model=model)
        to_save = {
            "predictions": predictions,
            "metrics": results,
            "gt": gt,
        }
        pkl.dump(to_save, open(f"./{args.model_key}_{name}.pkl", "wb"))
