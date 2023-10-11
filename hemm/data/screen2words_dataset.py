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

class Screen2WordsDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./',
                 kaggle_api_path = None
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.kaggle_api_path = kaggle_api_path
        self.prompt = Screen2WordsPrompt()

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
                         metric
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric
        
        predictions = []
        ground_truth = []
        
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
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
        
        results = self.metric.compute(ground_truth, predictions)
        return results

    def evaluate_dataset_batched(self,
                         model,
                         metric,
                         batch_size=32
                         ):
        self.load()
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
        
        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.chat.device)
        predictions = self.model.generate_batch(images_tensor, texts, batch_size)
        
        results = self.metric.compute(ground_truth, predictions)
        return results