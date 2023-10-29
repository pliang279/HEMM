import os
import json
from glob import glob
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.mmimdb_prompt import MMIMDBPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class MMIMDBDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/mmimdb/dataset',
                 annotation_file="/work/agoindan/mmimdb/split.json",
                 split="test",
                 device="cpu",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = MMIMDBPrompt()
        self.annotation_file = annotation_file
        self.split = split
        self.metrics = [BertScoreMetric(), BleuMetric()]

    # def load(self):
    #   os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
    #   if not os.path.exists('landuse-scene-classification.zip'):
    #       shell_command('kaggle datasets download -d apollo2506/landuse-scene-classification')
    #   if not os.path.exists('ucmercedimages'):
    #       shell_command('unzip landuse-scene-classification.zip -d ucmercedimages/')

    def load(self):
        pass

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []

        ann_files = json.load(open(self.annotation_file))[self.split.strip()]
        idx = 0
        for row in tqdm(ann_files, total=len(ann_files)):
            if idx == 100:
                break
            ann_id = row.strip()
            image_path = f"{self.image_dir}/{ann_id}.jpeg"
            data = json.load(open(f"{self.image_dir}/{ann_id}.json"))
            text = self.get_prompt(data["plot"][0])
            label = ", ".join(data["genres"])
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)
            idx += 1
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        
        return predictions, results
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []

        texts = []
        images = []

        ann_files = json.load(open(self.annotation_file))[self.split.strip()]
        idx = 0
        for row in tqdm(ann_files, total=len(ann_files)):
            if idx == 100:
                break
            ann_id = row.strip()
            image_path = f"{self.image_dir}/{ann_id}.jpeg"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            data = json.load(open(f"{self.image_dir}/{ann_id}.json"))
            text = self.get_prompt(data["plot"][0])
            label = ", ".join(data["genres"])
            texts.append(text)
            ground_truth.append(label)
            idx += 1
        
        predictions = self.predict_batched(images, texts, batch_size)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results
        