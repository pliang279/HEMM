import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, '..'))
sys.path.append(project_root)

os.chdir("/work/agoindan/.cache")
os.environ['TRANSFORMERS_CACHE'] = "/work/agoindan/.cache"
print(os.getenv("TRANSFORMERS_CACHE"))

from hemm.utils.base_utils import load_model, load_dataset_evaluator

model_key = 'instruct_blip'
model = load_model(model_key)
model.load_weights()

print("Model Loaded")

dataset_name = 'hateful_memes'
dataset_evaluator = load_dataset_evaluator(dataset_name, kaggle_api_path="/home/agoindan/")

print("Dataset Loaded")

# results = dataset_evaluator.evaluate_dataset(model=model)
results = dataset_evaluator.evaluate_dataset_batched(model=model)
print(results)
