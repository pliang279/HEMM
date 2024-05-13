from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
from hemm.prompts.nlvr_prompt import nlvrprompt
import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
from tqdm import tqdm
import random

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command

class NLVRDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='nlvr/nlvr/dev/',
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = nlvrprompt()

        self.load()
        self.image_dir = os.path.join(self.dataset_dir, 'images/')
        with open(os.path.join(self.dataset_dir, 'dev.json'), "r") as f:
            self.sentences = f.readlines()

    def load(self):
        shell_command(f'git clone https://github.com/lil-lab/nlvr.git {self.download_dir}/nlvr')

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model=None,
                         ) -> None:
        predictions = []
        ground_truth = []
        outputs = []        

        raw_images = []
        texts = []
        

        for line in tqdm(self.sentences, total=len(self.sentences)):
            ann = json.loads(line)
            img_path = os.path.join(self.image_dir, f'{ann["directory"]}/dev-{ann["identifier"]}-0.png')
            sentence = ann['sentence']
            text = self.get_prompt(sentence)
            
            output = model.generate(text, img_path)
            
            outputs.append(output)
            
            label = ann['label']
            ground_truth.append(label.lower())
            
        

        return outputs, ground_truth
    
    def evaluate_dataset_batched(self, model=None, batch_size=None):
        pass
