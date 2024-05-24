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
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = Screen2WordsPrompt()
        self.images_dir = os.path.join(self.dataset_dir, 'screen2wordsimages/unique_uis/combined')
        self.csv_path = os.path.join(self.dataset_dir, 'screen2words/screen_summaries.csv')
        self.test_file = os.path.join(self.dataset_dir, 'screen2words/split/test_screens.txt')
        self.load()

    def __len__(self):
        data_file = open(self.test_file, 'r')
        return len(data_file)        

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.dataset_dir}/rico-dataset.zip'):
            shell_command(f'kaggle datasets download -d onurgunes1993/rico-dataset -p {self.dataset_dir}')
        if not os.path.exists(f'{self.dataset_dir}/screen2wordsimages'):
            shell_command(f'unzip {self.dataset_dir}/rico-dataset.zip -d {self.dataset_dir}/screen2wordsimages')
        if not os.path.exists(f'{self.dataset_dir}/screen2words'):
            shell_command(f'git clone https://github.com/google-research-datasets/screen2words {self.dataset_dir}/screen2words')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def get_ground_truth(self, filename):
        ground_truth = list(self.dataset[self.dataset["screenId"] == int(filename)]["summary"])[-1]
        return ground_truth

    def evaluate_dataset(self,
                         model,
                         ) -> None:

        predictions = []
        ground_truth = []

        self.dataset = pd.read_csv(self.csv_path)

        data_file = open(self.test_file, 'r')
        data_lines = data_file.readlines()
        
        for line in tqdm(data_lines, total=len(data_lines)):
            file_name = line.strip()
            image_path = os.path.join(self.images_dir, file_name + '.jpg')
            ground_truth_answer = self.get_ground_truth(file_name)
            text = self.get_prompt()
            
            output = model.generate(text, image_path)
            
            predictions.append(output)
            ground_truth.append(ground_truth_answer)

        return predictions, ground_truth

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):

        self.model = model
        
        predictions = []
        ground_truth = []
        
        texts = []
        images = []

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
        
        predictions = self.predict_batched(images, texts, batch_size)
            
        return predictions, ground_truth