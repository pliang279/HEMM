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

class UCMercedDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./',
                 kaggle_api_path = None
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.kaggle_api_path = kaggle_api_path
        self.prompt = UCMercedPrompt()

    def load(self):
      os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
      if not os.path.exists('landuse-scene-classification.zip'):
          shell_command('kaggle datasets download -d apollo2506/landuse-scene-classification')
      if not os.path.exists('ucmercedimages'):
          shell_command('unzip landuse-scene-classification.zip -d ucmercedimages/')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         metric
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric
        
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
        
        results = self.metric.compute(ground_truth, predictions)
        return results