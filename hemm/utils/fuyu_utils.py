from hemm.metrics import accuracy_metric, bleu_metric, bertscore_metric
from hemm.models import fuyu
from hemm.data import newyorkercartoon_dataset, hateful_memes_dataset, nocaps_dataset, \
                        memotion_dataset, memecaps_dataset, irfl_dataset, scienceQA_dataset, \
                        vqa_dataset,vcr_dataset, ok_vqa_dataset, gqa_dataset, vqarad_dataset, \
                        pmcvqa_dataset, pathvqa_dataset, rsicd_dataset, ucmerced_dataset, \
                        resisc45_dataset, winogroundVQA_dataset, nlvr2, nlvr_dataset, faceemotion_dataset, \
                        cocoqa_dataset, visualgenome_dataset, screen2words_dataset, flickr30k_dataset, \
                        decimer_dataset, enrico_dataset, inat_dataset, magic_brush_dataset, mmimdb_dataset, \
                        plip_kather_dataset, slake_dataset
                        
def load_model(model_key):
    model_dict = {
        'fuyu': fuyu.Fuyu(),
    }
    return model_dict[model_key]

def load_metric(metric_key):
    metric_dict = {
        'accuracy':accuracy_metric.AccuracyMetric(),
        'bleu_score':bleu_metric.BleuMetric(),
        'bertscore':bertscore_metric.BertScoreMetric(),
        'cider':cider_metric.CiderMetric()
    }
    return metric_dict[metric_key]

def load_dataset_evaluator(kaggle_api_path=None):
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
        'vqa': vqa_dataset.VQADatasetEvaluator(),
        'vcr': vcr_dataset.VCRDatasetEvaluator(),
        'okvqa': ok_vqa_dataset.OKVQADatasetEvaluator(dataset_dir='/work/agoindan/.cache/'),
        'gqa': gqa_dataset.GQADatasetEvaluator(),
        'winogroundVQA': winogroundVQA_dataset.WinogroundDatasetEvaluator(),
        'nlvr':nlvr_dataset.NLVRDatasetEvaluator(),
        'nlvr2':nlvr2.NLVR2evaluator(),
        'visualgen':visualgenome_dataset.VisualGenomeEvaluator(),
        'vqarad': vqarad_dataset.VQARADDatasetEvaluator(),
        'pathvqa': pathvqa_dataset.PathVQADatasetEvaluator(),
        'ucmerced':ucmerced_dataset.UCMercedDatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
        'resisc45':resisc45_dataset.Resisc45DatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
        'face_emotion': faceemotion_dataset.FaceEmotionDatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
        'screen2words':screen2words_dataset.Screen2WordsDatasetEvaluator(
            kaggle_api_path=kaggle_api_path
        ),
        'decimer':decimer_dataset.DecimerDatasetEvaluator(),
        'slake':slake_dataset.SlakeDatasetEvaluator(),
        'enrico': enrico_dataset.EnricoDatasetEvaluator(),
        'flickr30k': flick30k_new_dataset.Flickr30kDatasetEvaluator(),
        'inat': inat_dataset.INATDatasetEvaluator(),
        # 'magic_brush': magic_brush_dataset.MagicBrushDatasetEvaluator(),
        'mmimdb': mmimdb_dataset.MMIMDBDatasetEvaluator(), 
        "plip": plip_kather_dataset.PlipKatherDatasetEvaluator(),
    }
    return dataset_dict
