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
from hemm.metrics.metric import HEMMMetric
from hemm.metrics.bertscore_metric import BertScoreMetric


class WinogroundDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./',
                 hf_auth_token="",
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.prompt = WinogroundPrompt()
        self.metrics = [BertScoreMetric()]
        self.hf_auth_token = hf_auth_token
        self.load()

    def load(self):
        self.dataset = load_dataset("facebook/winoground", use_auth_token=self.hf_auth_token)['test']

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.model = model
        
        predictions = []
        ground_truth = []
        for data in tqdm(self.dataset, total=len(self.dataset)):
            for pair in ((0,0), (0,1), (1,0), (1,1)):
                image = data['image_'+str(pair[0])]
                query = data['caption_'+str(pair[1])]
                question=self.get_prompt(query)
                gt = 'yes' if pair[0] == pair[1] else 'no'
                ground_truth.append(gt)
                with open("current_image.jpg", 'wb') as f:
                # f.write(image_url)
                    image = image.convert('RGB')
                    image.save(f)
                    image_path = "current_image.jpg"
                output = self.model.generate(question, image_path)
                predictions.append(output) 

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)

        return predictions, results

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):
        pass
