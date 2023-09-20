from hemm.utils.base_utils import load_model, load_dataset_evaluator, load_metric

model_key = 'minigpt4'
model = load_model(model_key)
model.load_weights()

dataset_name = 'winogroundVQA'
dataset_evaluator = load_dataset_evaluator(dataset_name)

metric_name = 'accuracy'
metric = load_metric(metric_name)

results = dataset_evaluator.evaluate_dataset(model=model, metric=metric)
print(results)
