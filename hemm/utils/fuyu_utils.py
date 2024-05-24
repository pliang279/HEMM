from hemm.metrics import accuracy_metric, bleu_metric, bertscore_metric
from hemm.models import fuyu
from hemm.data import newyorkercartoon_dataset, hateful_memes_dataset, nocaps_dataset, \
                        memotion_dataset, memecaps_dataset, irfl_dataset, scienceQA_dataset, \
                        vqa_dataset,vcr_dataset, ok_vqa_dataset, gqa_dataset, vqarad_dataset, \
                        pathvqa_dataset, ucmerced_dataset, resisc45_dataset, winogroundVQA_dataset, \
                        nlvr2, nlvr_dataset, faceemotion_dataset, \
                        visualgenome_dataset, screen2words_dataset, flickr30k_dataset, \
                        decimer_dataset, enrico_dataset, inat_dataset, magic_brush_dataset, mmimdb_dataset, \
                        open_path_dataset, slake_dataset, lncoco_dataset
                        
def load_model(model_key, download_dir="./", **kwargs):
    model_dict = {
        'fuyu': fuyu.Fuyu,
    }
    model = model_dict[model_key](download_dir=download_dir, **kwargs)
    return model

def load_metric(metric_key):
    metric_dict = {
        'accuracy':accuracy_metric.AccuracyMetric(),
        'bleu_score':bleu_metric.BleuMetric(),
        'bertscore':bertscore_metric.BertScoreMetric(),
        'cider':cider_metric.CiderMetric()
    }
    return metric_dict[metric_key]

def load_dataset_evaluator(dataset_key, download_dir="./", kaggle_api_path=None, hf_auth_token=None):
    dataset_dict = {
        'hateful_memes':hateful_memes_dataset.HatefulMemesDatasetEvaluator,
        'newyorkercartoon':newyorkercartoon_dataset.NewYorkerCartoonDatasetEvaluator,
        'nocaps':nocaps_dataset.NoCapsDatasetEvaluator,
        'memotion':memotion_dataset.MemotionDatasetEvaluator,
        'memecaps':memecaps_dataset.MemeCapsDatasetEvaluator,
        'irfl':irfl_dataset.IRFLDatasetEvaluator,
        'scienceqa':scienceQA_dataset.ScienceQADatasetEvaluator,
        'vqa': vqa_dataset.VQADatasetEvaluator,
        'vcr': vcr_dataset.VCRDatasetEvaluator,
        'okvqa': ok_vqa_dataset.OKVQADatasetEvaluator,
        'gqa': gqa_dataset.GQADatasetEvaluator,
        'winogroundVQA': winogroundVQA_dataset.WinogroundDatasetEvaluator,
        'nlvr':nlvr_dataset.NLVRDatasetEvaluator,
        'nlvr2':nlvr2.NLVR2evaluator,
        'visualgen':visualgenome_dataset.VisualGenomeEvaluator,
        'vqarad': vqarad_dataset.VQARADDatasetEvaluator,
        'pathvqa': pathvqa_dataset.PathVQADatasetEvaluator,
        'ucmerced':ucmerced_dataset.UCMercedDatasetEvaluator,
        'resisc45':resisc45_dataset.Resisc45DatasetEvaluator,
        'face_emotion': faceemotion_dataset.FaceEmotionDatasetEvaluator,
        'screen2words':screen2words_dataset.Screen2WordsDatasetEvaluator,
        'decimer':decimer_dataset.DecimerDatasetEvaluator,
        'slake':slake_dataset.SlakeDatasetEvaluator,
        'enrico': enrico_dataset.EnricoDatasetEvaluator,
        'flickr30k': flickr30k_dataset.Flickr30kDatasetEvaluator,
        'inat': inat_dataset.INATDatasetEvaluator,
        'magic_brush': magic_brush_dataset.MagicBrushDatasetEvaluator,
        'mmimdb': mmimdb_dataset.MMIMDBDatasetEvaluator, 
        "open_path": open_path_dataset.OpenPathDatasetEvaluator,
        'lncoco': lncoco_dataset.LNCOCODatasetEvaluator,
    }
    dset = dataset_dict[dataset_key](
        download_dir=download_dir,
        kaggle_api_path=kaggle_api_path,
        hf_auth_token=hf_auth_token,
    )
    return dset
