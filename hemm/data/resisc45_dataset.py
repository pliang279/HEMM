import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import random

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.resisc45_prompt import Resisc45Prompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class Resisc45DatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./',
                 kaggle_api_path = None
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.kaggle_api_path = kaggle_api_path
        self.prompt = Resisc45Prompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists('nwpu-data-set.zip'):
            shell_command('kaggle datasets download -d happyyang/nwpu-data-set')
        if not os.path.exists('resisc45'):
            shell_command('unzip nwpu-data-set.zip -d resisc45')
        
        self.images_dir = 'resisc45/NWPU Data Set/NWPU-RESISC45/NWPU-RESISC45'
        classes = os.listdir(self.images_dir)

        images_list = []
        ground_truth_list = []
        for image_class in classes:
            x = os.listdir(os.path.join(self.images_dir, image_class))
            images_list.extend(x)
            ground_truth_list.extend([image_class for i in range(len(x))])

        data_list = []
        for x,y in zip(images_list, ground_truth_list):
            data_list.append((x, y))

        random.shuffle(data_list)
        self.dataset = data_list

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []
        
        
        for data in tqdm(self.dataset, total=len(self.dataset)):
            image_path = os.path.join(self.images_dir, data[1], data[0])
            ground_truth_answer = data[1]
            text = self.get_prompt()
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
        
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
        
        predictions = []
        ground_truth = []
        
        texts = []
        images = []
        
        for data in tqdm(self.dataset, total=len(self.dataset)):
            image_path = os.path.join(self.images_dir, data[1], data[0])
            ground_truth_answer = data[1]
            text = self.get_prompt()
            texts.append(text)
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            ground_truth.append(ground_truth_answer)
        
        samples = len(images) // 10
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth[:samples], predictions)
        
        return predictions, results, ground_truth[:samples]
    