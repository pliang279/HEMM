import subprocess

from hemm.metrics import accuracy_metric, bleu_metric
from hemm.models import blip2_model, minigpt4_model
from hemm.data import newyorkercartoon_dataset, hateful_memes_dataset, nocaps_dataset, memotion_dataset, memecaps_dataset, irfl_dataset, scienceQA_dataset, vqa_dataset,vcr_dataset, ok_vqa_dataset, gqa_dataset, vqarad_dataset, pmcvqa_dataset, pathvqa_dataset, rsicd_dataset

def load_model(model_key):
    model_dict = {
        'minigpt4': minigpt4_model.MiniGPT4(),
        'blip2':blip2_model.BLIP2(model_type='pretrain_flant5xxl')
    }
    return model_dict[model_key]

def load_metric(metric_key):
    metric_dict = {
        'accuracy':accuracy_metric.AccuracyMetric(),
        'bleu_score':bleu_metric.BleuMetric()
    }
    return metric_dict[metric_key]

def load_dataset_evaluator(dataset_key, kaggle_api_path=None):
    dataset_dict = {
        'hateful_memes':hateful_memes_dataset.HatefulMemesDatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
        'newyorkercartoon':newyorkercartoon_dataset.NewYorkerCartoonDatasetEvaluator(),
        'nocaps':nocaps_dataset.NoCapsDatasetEvaluator(),
        'memotion':memotion_dataset.MemotionDatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
        'memecaps':memecaps_dataset.MemeCapsDatasetEvaluator(),
        'irfl':irfl_dataset.IRFLDatasetEvaluator(),
        'scienceqa':scienceQA_dataset.ScienceQADatasetEvaluator(),
        'vcr': vqa_dataset.VQADatasetEvaluator(),
        'vqa': vcr_dataset.VCRDatasetEvaluator(),
        'okvqa': ok_vqa_dataset.OKVQADatasetEvaluator(),
        'gqa': gqa_dataset.GQADatasetEvaluator(),
        'vqarad': vqarad_dataset.VQARADDatasetEvaluator(),
        'pmcvqa': pmcvqa_dataset.PMCVQADatasetEvaluator(),
        'pathvqa': pathvqa_dataset.PathVQADatasetEvaluator(),
        'rsicd': rsicd_dataset.RSICDDatasetEvaluator()
    }
    return dataset_dict[dataset_key]