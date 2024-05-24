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
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = UCMercedPrompt()
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.dataset_dir}/landuse-scene-classification.zip'):
            shell_command(f'kaggle datasets download -d apollo2506/landuse-scene-classification -P {self.dataset_dir}')
        if not os.path.exists(f'{self.dataset_dir}/ucmercedimages'):
            shell_command(f'unzip {self.dataset_dir}/landuse-scene-classification.zip -d {self.dataset_dir}/ucmercedimages/')

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self):
        csv_path = f'{self.dataset_dir}/ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        return len(df)

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        
        predictions = []
        ground_truth = []

        csv_path = f'{self.dataset_dir}/ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        images_dir = f'{self.dataset_dir}/ucmercedimages/images_train_test_val/validation'

        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(images_dir, row['Filename'])
            ground_truth_answer = row['ClassName']
            text = self.get_prompt()
            
            output = model.generate(text, image_path)
            
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
         
        return predictions, ground_truth
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.model = model
        
        predictions = []
        ground_truth = []

        texts = []
        images = []

        csv_path = f'{self.dataset_dir}/ucmercedimages/validation.csv'
        df = pd.read_csv(csv_path)
        images_dir = f'{self.dataset_dir}/ucmercedimages/images_train_test_val/validation'

        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(images_dir, row['Filename'])
            ground_truth_answer = row['ClassName']
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(ground_truth_answer)
        
        predictions = self.predict_batched(images, texts, batch_size)

        return predictions, ground_truth
    