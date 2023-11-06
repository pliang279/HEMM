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
from hemm.prompts.screen2words_prompt import Screen2WordsPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class Screen2WordsDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./',
                 kaggle_api_path = None
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.kaggle_api_path = kaggle_api_path
        self.prompt = Screen2WordsPrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]
        self.images_dir = 'screen2wordsimages/unique_uis/combined'
        self.csv_path = 'screen2words/screen_summaries.csv'
        self.test_file = 'screen2words/split/test_screens.txt'
        self.load()

    def __len__(self):
        data_file = open(self.test_file, 'r')
        return len(data_file)        

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists('rico-dataset.zip'):
            shell_command('kaggle datasets download -d onurgunes1993/rico-dataset')
        if not os.path.exists('screen2wordsimages'):
            shell_command('unzip rico-dataset.zip -d screen2wordsimages')
        if not os.path.exists('screen2words'):
            shell_command('git clone https://github.com/google-research-datasets/screen2words')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def get_ground_truth(self, filename):
        ground_truth = self.dataset.loc[self.dataset['screenId'] == filename]['summary']
        return ground_truth

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        # self.load()
        self.model = model

        predictions = []
        ground_truth = []

        self.dataset = pd.read_csv(self.csv_path)

        data_file = open(self.test_file, 'r')
        data_lines = data_file.readlines()

        for line in data_lines:
            file_name = line.strip()
            image_path = os.path.join(self.images_dir, file_name + '.jpg')
            ground_truth_answer = self.get_ground_truth(file_name)
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
        # self.load()
        self.model = model
        self.metric = metric
        
        predictions = []
        ground_truth = []
        
        texts = []
        images = []

        self.images_dir = 'screen2wordsimages/unique_uis/combined'
        self.csv_path = 'screen2words/screen_summaries.csv'
        self.test_file = 'screen2words/split/test_screens.txt'

        self.dataset = pd.read_csv(self.csv_path)

        data_file = open(self.test_file, 'r')
        data_lines = data_file.readlines()

        for line in data_lines:
            file_name = line.strip()
            image_path = os.path.join(self.images_dir, file_name + '.jpg')
            ground_truth_answer = self.get_ground_truth(file_name)
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