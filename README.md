
# HEMM

Create a virtual environment and install dependencies.

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cd HEMM
```

Run the code as follows 

```
python -m hemm.main
```

Note: For some datasets(Hateful memes, Memotion), kaggle is used. Make sure to get your api key from the following [link](https://github.com/Kaggle/kaggle-api).
Provide the path of the directory where kaggle.json is stored in the ```load_dataset_evaluator```

Sample code:

```python
from hemm.utils.base_utils import load_model, load_dataset_evaluator, load_metric

model_key = 'minigpt4'
model = load_model(model_key)
model.load_weights()

dataset_name = 'hateful_memes'
dataset_evaluator = load_dataset_evaluator(dataset_name, kaggle_api_directory='./')

metric_name = 'accuracy'
metric = load_metric(metric_name)

## For single data point evaluation
results = dataset_evaluator.evaluate_dataset(model=model, metric=metric)
print(results)

## For batching evaluation
results = dataset_evaluator.evaluate_dataset_batched(model=model, metric=metric, batch_size=32)
print(results)
```


---------------------------------------------------------------------------


Following models and datasets are supported -> 

Models
```
blip2
minigpt4
```

Datasets
```
hateful_memes
newyorkercartoon
memecaps
memotion
nocaps
irfl
scienceqa
vqa
vcr
gqa
okvqa
vqarad
resisc45
ucmerced
pathvqa
```

To evaluate these datasets, metrics are specified in the ```hemm.metrics``` directory
For ```memecaps``` and ```nocaps``` dataset, ```bleu_score``` or ```bertscore``` metric is used. All QA datasets use ```bertscore``` metric. Rest of the other datasets use ```accuracy``` metric. 


-----------------------------------------------------------------------------------------

To add new datasets, metrics and models, base class is provided in each of the modules ```hemm/models/model.py```, ```hemm/metrics/metric.py``` and ```hemm/data/dataset.py``` Inheriting these abstract classess will allow the user to contribute to HEMM.
