from hemm.metrics import accuracy_metric, bleu_metric, bertscore_metric
from hemm.models import blip2_model, minigpt4_model, instruct_blip, kosmos2, \
                         openflamingo_model, llama_adapter_model, emu_model, mplugowl_model
from hemm.data import newyorkercartoon_dataset, hateful_memes_dataset, nocaps_dataset, \
                        memotion_dataset, memecaps_dataset, irfl_dataset, scienceQA_dataset, \
                        vqa_dataset,vcr_dataset, ok_vqa_dataset, gqa_dataset, vqarad_dataset, \
                        pathvqa_dataset, ucmerced_dataset, resisc45_dataset, winogroundVQA_dataset, \
                        nlvr2, nlvr_dataset, faceemotion_dataset, \
                        visualgenome_dataset, screen2words_dataset, flickr30k_dataset, \
                        decimer_dataset, enrico_dataset, inat_dataset, magic_brush_dataset, mmimdb_dataset, \
                        open_path_dataset, slake_dataset
                        
def load_model(model_key, download_dir="./", **kwargs):
    if "llama_dir" in kwargs:
        llama_dir = kwargs["llama_dir"]
    model_dict = {
        'minigpt4': minigpt4_model.MiniGPT4(download_dir=download_dir),
        'blip2':blip2_model.BLIP2(model_type='pretrain_flant5xxl'),
        'instruct_blip':instruct_blip.InstructBlip(model_type="flant5xl"),
        'kosmos2': kosmos2.Kosmos2(),
        'openflamingo': openflamingo_model.OpenFlamingoModel(cache_dir=download_dir),
        'llama_adapter': llama_adapter_model.LlamaAdapter(llama_dir=llama_dir, download_dir=f"{download_dir}/llama_adapter_weights/"),
        'emu': emu_model.EmuModel(download_dir="{download_dir}/emu/"),
        'mplugowl': mplugowl_model.MplugOwl(),
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

def load_dataset_evaluator(dataset_key, download_dir="./", kaggle_api_path=None, hf_auth_token=None):
    dataset_dict = {
        'hateful_memes':hateful_memes_dataset.HatefulMemesDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token,
        ),
        'newyorkercartoon':newyorkercartoon_dataset.NewYorkerCartoonDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        
        'nocaps':nocaps_dataset.NoCapsDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token,
        ),
        'memotion':memotion_dataset.MemotionDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token
        ),
        'memecaps':memecaps_dataset.MemeCapsDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'irfl':irfl_dataset.IRFLDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'scienceqa':scienceQA_dataset.ScienceQADatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'vqa': vqa_dataset.VQADatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'vcr': vcr_dataset.VCRDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'okvqa': ok_vqa_dataset.OKVQADatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'gqa': gqa_dataset.GQADatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'winogroundVQA': winogroundVQA_dataset.WinogroundDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'nlvr':nlvr_dataset.NLVRDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'nlvr2':nlvr2.NLVR2evaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'visualgen':visualgenome_dataset.VisualGenomeEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'vqarad': vqarad_dataset.VQARADDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'pathvqa': pathvqa_dataset.PathVQADatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'ucmerced':ucmerced_dataset.UCMercedDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'resisc45':resisc45_dataset.Resisc45DatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token
        ),
        'face_emotion': faceemotion_dataset.FaceEmotionDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token
        ),
        'screen2words':screen2words_dataset.Screen2WordsDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token        
            ),
        'decimer':decimer_dataset.DecimerDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'slake':slake_dataset.SlakeDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'enrico': enrico_dataset.EnricoDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'flickr30k': flickr30k_dataset.Flickr30kDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'inat': inat_dataset.INATDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'magic_brush': magic_brush_dataset.MagicBrushDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
        'mmimdb': mmimdb_dataset.MMIMDBDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token), 
        "plip": open_path_dataset.PlipKatherDatasetEvaluator(
            download_dir=download_dir,
            kaggle_api_path=kaggle_api_path,
            hf_auth_token=hf_auth_token),
    }
    return dataset_dict[dataset_key]
