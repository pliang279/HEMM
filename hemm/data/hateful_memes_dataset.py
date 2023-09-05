import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import subprocess

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.hateful_memes_prompt import HatefulMemesPrompt
from hemm.utils.common_utils import shell_command

class HatefulMemesDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir = 'hateful_memes',
                 evaluate_path = 'dev.jsonl',
                 kaggle_api_path = None
                 ):
        super().__init__()

        self.dataset_key = 'hateful_memes'
        self.dataset_dir = dataset_dir
        self.evaluate_path = evaluate_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kaggle_api_path = kaggle_api_path
        self.prompt = HatefulMemesPrompt()
    
    def load(self, kaggle_api_path):
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_api_path
        if not os.path.exists('facebook-hateful-meme-dataset.zip'):
          shell_command('kaggle datasets download -d parthplc/facebook-hateful-meme-dataset')
        if not os.path.exists('hateful_memes'):
          shell_command('unzip facebook-hateful-meme-dataset.zip -d hateful_memes/')

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.model = model
        self.metric = metric
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data')

        ground_truth = []
        predictions = []
        for index in tqdm(range(len(json_list)), total=len(json_list)):
            json_obj = json.loads(json_list[index])
            text = self.get_prompt(json_obj['text'])
            output = self.model.generate(text, os.path.join(image_dir, json_obj['img']))
            answer = self.model.answer_extractor(output, self.dataset_key)
            if answer == 'yes':
                predictions.append(1)
            else:
                predictions.append(0)
            ground_truth.append(json_obj['label'])

        results = self.metric.compute(ground_truth, predictions)
        return results
    
    def evaluate_dataset_batched(self,
                         model,
                         metric,
                         batch_size=32
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.model = model
        self.metric = metric
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data')

        ground_truth = []
        predictions = []

        images = []
        texts = []

        for index in tqdm(range(len(json_list)), total=len(json_list)):
            json_obj = json.loads(json_list[index])
            text = self.get_prompt(json_obj['text'])
            image_path = os.path.join(image_dir, json_obj['img'])
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            texts.append(text)
            ground_truth.append(json_obj['label'])

        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.chat.device)
        outputs = self.model.generate_batch(images_tensor, texts, batch_size)
        for output in outputs:
            answer = self.model.answer_extractor(output, self.dataset_key)
            if answer == 'yes':
                predictions.append(1)
            else:
                predictions.append(0)
        results = self.metric.compute(ground_truth, predictions)
        return results