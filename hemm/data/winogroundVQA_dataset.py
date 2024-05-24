import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from hemm.prompts.winoground_prompt import WinogroundPrompt
from hemm.data.dataset import HEMMDatasetEvaluator


class WinogroundDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.prompt = WinogroundPrompt()
        self.hf_auth_token = kwargs["hf_auth_token"]
        self.load()
    
    def load(self):
        self.dataset = load_dataset("facebook/winoground", use_auth_token=self.hf_auth_token, cache_dir=self.dataset_dir)['test']

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        predictions = []
        ground_truth = []
        texts = []
        total_samples = 4 * len(self.dataset)
        
        for data in tqdm(self.dataset, total=len(self.dataset)):
            for pair in ((0,0), (0,1), (1,0), (1,1)):
                image = data['image_'+str(pair[0])]
                query = data['caption_'+str(pair[1])]
                question=self.get_prompt(query)
                gt = 'yes' if pair[0] == pair[1] else 'no'
                ground_truth.append(gt)
                with open("current_image.jpg", 'wb') as f:
                    image = image.convert('RGB')
                    image.save(f)
                    image_path = "current_image.jpg"
                texts.append(question)
                
                output = model.generate(question, image_path)
                
                predictions.append(output)

        return predictions, ground_truth

    def evaluate_dataset_batched(self,
                         model=None,
                         batch_size=32
                         ):
        pass
