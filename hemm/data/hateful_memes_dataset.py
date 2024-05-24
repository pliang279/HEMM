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
                download_dir="./",
                dataset_dir='hateful_memes',
                annotation_file='dev.jsonl',
                **kwargs,
                 ):
        super().__init__()
        self.download_dir = download_dir
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.evaluate_path = annotation_file
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = HatefulMemesPrompt()
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.download_dir}/facebook-hateful-meme-dataset.zip'):
          shell_command(f'kaggle datasets download -d parthplc/facebook-hateful-meme-dataset -p {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/hateful_memes'):
          shell_command(f'unzip {self.download_dir}/facebook-hateful-meme-dataset.zip -d {self.download_dir}/hateful_memes/')
    
    def __len__(self):
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        return len(json_list)

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
    
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data')

        ground_truth = []
        predictions = []
        outputs = []
        for index in tqdm(range(len(json_list)), total=len(json_list)):
            json_obj = json.loads(json_list[index])
            text = self.get_prompt(json_obj['text'])
            output = model.generate(text, os.path.join(image_dir, json_obj['img']))

            outputs.append(output)
            if json_obj["label"]:
                ground_truth.append("yes")
            else:
                ground_truth.append("no")
            

        return outputs, ground_truth
     
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        
        self.model = model
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
            image = model.get_image_tensor(raw_image)
            images.append(image)
            texts.append(text)
            if json_obj["label"]:
                ground_truth.append("yes")
            else:
                ground_truth.append("no")

        outputs = self.predict_batched(images, texts, batch_size)
        return outputs, ground_truth
