import os
import json
from PIL import Image
import requests
import torch
import subprocess
from tqdm import tqdm
import pandas as pd
import random 

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.prompts.memotion_prompt import MemotionPrompt

class MemotionDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 data_path = './memotion_dataset_7k/labels.xlsx',
                 image_dir = './memotion_dataset_7k/images',
                 kaggle_api_path = None
                 ):
        super().__init__()
        self.dataset_key = 'memotion'
        self.data_path = data_path
        self.image_dir = image_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kaggle_api_path = kaggle_api_path
        self.prompt = MemotionPrompt()
        self.choices = ['funny', 'very_funny', 'not_funny', 'hilarious']

    def get_prompt(self, caption) -> str:
        prompt_text = self.prompt.format_prompt(caption)
        return prompt_text

    def load(self, kaggle_api_path):
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_api_path
        subprocess.Popen('kaggle datasets download -d williamscott701/memotion-dataset-7k', shell=True)
        subprocess.Popen('unzip archive.zip -d ./', shell=True)

    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.metric = metric
        self.model = model
        self.model.to(self.device)
        df = pd.read_excel(self.data_path)
        predictions = []
        ground_truth = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(self.image_dir, row['image_name'])
            caption = row['text_corrected']
            gt_label = row['humour']
            ground_truth.append(self.choices.index(gt_label))
            text = self.get_prompt(caption)
            output = self.model.generate(image_path, text)
            answer = self.model.answer_extractor(output, self.dataset_key)
            if answer:
                predictions.append(answer)
            else:
                random_item = random.choice(list(range(0, 4)))
                predictions.append(random_item)
        
        results = self.metric(ground_truth, predictions)
        return results