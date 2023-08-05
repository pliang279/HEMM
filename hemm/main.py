from hemm.utils.base_utils import load_model, load_dataset_evaluator, load_metric, evaluate_dataset

model_key = 'minigpt4'
model = load_model(model_key)

dataset_name = 'hateful_memes'
dataset_evaluator = load_dataset_evaluator(dataset_name)

metric_name = 'accuracy'
metric = load_metric(metric_name)

results = dataset_evaluator.evaluate(model=model, metric=metric)
print(results)