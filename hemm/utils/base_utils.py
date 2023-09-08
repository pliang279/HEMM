import subprocess

from hemm.metrics import accuracy_metric, bleu_metric, bertscore_metric
from hemm.models import blip2_model, minigpt4_model
from hemm.data import newyorkercartoon_dataset, hateful_memes_dataset, nocaps_dataset, memotion_dataset, memecaps_dataset, irfl_dataset, scienceQA_dataset, vqa_dataset,vcr_dataset, ok_vqa_dataset, gqa_dataset, vqarad_dataset, pmcvqa_dataset, pathvqa_dataset, rsicd_dataset, ucmerced_dataset, resisc45_dataset,winogroundVQA_dataset,winoground_dataset,nlvr2,nlvr_dataset

def load_model(model_key):
    model_dict = {
        'minigpt4': minigpt4_model.MiniGPT4(),
        'blip2':blip2_model.BLIP2(model_type='pretrain_flant5xxl')
    }
    return model_dict[model_key]

def load_metric(metric_key):
    metric_dict = {
        'accuracy':accuracy_metric.AccuracyMetric(),
        'bleu_score':bleu_metric.BleuMetric(),
        'bertscore':bertscore_metric.BertScoreMetric()
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
        'winogroundVQA': winogroundVQA_dataset.WinogroundVQAEvaluator(),
        'nlvr':nlvr_dataset.NLVRDatasetEvaluator(),
        'nlvr2':nlvr2.nlvr2evaluator(),
        'vqarad': vqarad_dataset.VQARADDatasetEvaluator(),
        'pathvqa': pathvqa_dataset.PathVQADatasetEvaluator(),
        'ucmerced':ucmerced_dataset.UCMercedDatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
        'resisc45':resisc45_dataset.Resisc45DatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
    }
    return dataset_dict[dataset_key]
