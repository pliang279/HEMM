import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import subprocess

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.hateful_memes_prompt import HatefulMemesPrompt
from hemm.utils.common_utils import shell_command
from hemm.metrics.accuracy_metric import *
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class HatefulMemesDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir = 'hateful_memes',
                 evaluate_path = 'dev.jsonl',
                 kaggle_api_path = None
                 ):
        super().__init__()

        self.dataset_key = 'hateful_memes'
        self.dataset_dir = dataset_dir
        self.evaluate_path = evaluate_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kaggle_api_path = kaggle_api_path
        self.prompt = HatefulMemesPrompt()
        self.metrics = [AccuracyMetric(), PrecisionMetric(), RecallMetric(), 
                        F1ScoreMetric()]
    
    def load(self, kaggle_api_path):
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_api_path
        if not os.path.exists('facebook-hateful-meme-dataset.zip'):
          shell_command('kaggle datasets download -d parthplc/facebook-hateful-meme-dataset')
        if not os.path.exists('hateful_memes'):
          shell_command('unzip facebook-hateful-meme-dataset.zip -d hateful_memes/')

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.model = model
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data')

        ground_truth = []
        predictions = []
        outputs = []
        for index in tqdm(range(len(json_list)), total=len(json_list)):
            json_obj = json.loads(json_list[index])
            text = self.get_prompt(json_obj['text'])
            output = self.model.generate(text, os.path.join(image_dir, json_obj['img']))
            outputs.append(output)
            answer = self.model.answer_extractor(output, self.dataset_key)
            if answer == 'yes':
                predictions.append(1)
            else:
                predictions.append(0)
            ground_truth.append(json_obj['label'])

        results = {}
        for metric in self.metrics:
            metric_val = metric.compute(ground_truth, predictions)
            results[metric.name] = metric_val
        return outputs, results
     
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.model = model
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data')

        ground_truth = []
        predictions = []

        images = []
        texts = []

        for index in tqdm(range(len(json_list)), total=len(json_list)):
            json_obj = json.loads(json_list[index])
            text = self.get_prompt(json_obj['text'])
            image_path = os.path.join(image_dir, json_obj['img'])
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            texts.append(text)
            ground_truth.append(json_obj['label'])

        outputs = self.predict_batched(images, texts, batch_size)

        for output in outputs:
            answer = self.model.answer_extractor(output, self.dataset_key)
            if answer == 'yes':
                predictions.append(1)
            else:
                predictions.append(0)
        
        results = {}
        for metric in self.metrics:
            metric_val = metric.compute(ground_truth, predictions)
            results[metric.name] = metric_val
        # results = self.metric.compute(ground_truth, predictions)
        return outputs, results
    