import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import subprocess
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.nocaps_prompt import NoCapsPrompt
from hemm.utils.common_utils import shell_command
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class NoCapsDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir = './nocaps_val_4500_captions.json',
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt = NoCapsPrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def load(self):
        shell_command('wget https://s3.amazonaws.com/nocaps/nocaps_val_4500_captions.json')

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        json_file = json.load(open(self.dataset_dir, 'r'))
        predictions = []
        ground_truth = []
        for index, image_dict in tqdm(enumerate(json_file['images']), total=len(json_file['images'])):
            image_url = image_dict['coco_url']
            image_caption = json_file['annotations'][image_dict['id']]['caption']
            text = self.get_prompt()
            response = requests.get(image_url)
            if response.status_code == 200:
                with open("./current_image.jpg", 'wb') as f:
                    f.write(response.content)
            image_path = "./current_image.jpg"
            ground_truth.append(image_caption)
            output = self.model.generate(text, image_path)
            predictions.append(output)
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results
 
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):
        self.load()
        self.model = model
        json_file = json.load(open(self.dataset_dir, 'r'))
        predictions = []
        ground_truth = []

        texts = []
        images = []

        for index, image_dict in tqdm(enumerate(json_file['images']), total=len(json_file['images'])):
            image_url = image_dict['coco_url']
            image_caption = json_file['annotations'][image_dict['id']]['caption']
            text = self.get_prompt()
            response = requests.get(image_url)
            if response.status_code == 200:
                with open("./current_image.jpg", 'wb') as f:
                    f.write(response.content)
            image_path = "./current_image.jpg"

            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            
            texts.append(text)
            ground_truth.append(image_caption)
        
        predictions = self.predict_batched(images, texts, batch_size)
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        
        return predictions, results
    