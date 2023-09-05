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
from hemm.utils.common_utils import shell_command

class MemotionDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 data_path = 'memotion-dataset-7k/memotion_dataset_7k/labels.xlsx',
                 image_dir = 'memotion-dataset-7k/memotion_dataset_7k/images',
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
        if not os.path.exists('memotion-dataset-7k.zip'):
          shell_command('kaggle datasets download -d williamscott701/memotion-dataset-7k')
        if not os.path.exists('memotion-dataset-7k'):
          shell_command('unzip memotion-dataset-7k.zip -d memotion-dataset-7k')

    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.metric = metric
        self.model = model
        df = pd.read_excel(self.data_path)
        predictions = []
        ground_truth = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(self.image_dir, row['image_name'])
            caption = row['text_corrected']
            gt_label = row['humour']
            ground_truth.append(self.choices.index(gt_label))
            text = self.get_prompt(caption)
            output = self.model.generate(text, image_path)
            answer = self.model.answer_extractor(output, self.dataset_key)
            if answer:
                predictions.append(answer)
            else:
                random_item = random.choice(list(range(0, 4)))
                predictions.append(random_item)
        
        results = self.metric.compute(ground_truth, predictions)
        return results

    def evaluate_dataset_batched(self,
                         model,
                         metric,
                         batch_size=32
                         ):
        self.load(self.kaggle_api_path)
        self.metric = metric
        self.model = model
        df = pd.read_excel(self.data_path)
        predictions = []
        ground_truth = []
        images = []
        texts = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(self.image_dir, row['image_name'])
            caption = row['text_corrected']
            gt_label = row['humour']
            ground_truth.append(self.choices.index(gt_label))

            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)

            text = self.get_prompt(caption)
            output = self.model.generate(text, image_path)
            answer = self.model.answer_extractor(output, self.dataset_key)
            texts.append(text)

        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.chat.device)
        outputs = self.model.generate_batch(images_tensor, texts, batch_size)
        for answer in outputs:
            if answer:
                predictions.append(answer)
            else:
                random_item = random.choice(list(range(0, 4)))
                predictions.append(random_item)
        
        results = self.metric.compute(ground_truth, predictions)
        return results