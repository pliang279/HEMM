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
                 dataset_dir = './',
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
        shell_command('kaggle datasets download -d parthplc/facebook-hateful-meme-dataset')
        shell_command('unzip archive.zip -d ./')

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
        image_dir = os.path.join(self.dataset_dir, 'data', 'img')

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