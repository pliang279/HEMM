import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.ucmerced_prompt import UCMercedPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class UCMercedDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./',
                 kaggle_api_path = None
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.kaggle_api_path = kaggle_api_path
        self.prompt = UCMercedPrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]

    def load(self):
      os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
      if not os.path.exists('landuse-scene-classification.zip'):
          shell_command('kaggle datasets download -d apollo2506/landuse-scene-classification')
      if not os.path.exists('ucmercedimages'):
          shell_command('unzip landuse-scene-classification.zip -d ucmercedimages/')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self):
        csv_path = 'ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        return len(df)

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []

        csv_path = 'ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        images_dir = 'ucmercedimages/images_train_test_val/validation'

        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(images_dir, row['Filename'])
            ground_truth_answer = row['ClassName']
            text = self.get_prompt()
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
    
        return predictions, ground_truth
    
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
        # raw_images = []

        csv_path = 'ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        images_dir = 'ucmercedimages/images_train_test_val/validation'

        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(images_dir, row['Filename'])
            ground_truth_answer = row['ClassName']
            raw_image = Image.open(image_path).convert('RGB')
            # raw_images.append(raw_image)
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(ground_truth_answer)
        
        samples = len(images)
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)
        # print(len(raw_images))
        # samples = len(raw_images)
        # self.save_details(raw_images[:samples], texts[:samples], ground_truth[:samples], "ucmerced.pkl")

        return predictions, ground_truth[:samples]
    