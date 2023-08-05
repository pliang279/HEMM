import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import pandas as pd
import ast

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.utils.evaluator_mixin import EvaluatorMixin
from hemm.prompts.irfl_prompt import IRFLPrompt


class IRFLPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 image_processor,
                 text_processor,
                 prompt,
                 device,
                 ):
        self.dataset = dataset
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.device = device
        self.prompt = prompt

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        phrase = row['phrase']
        img_path = row['image']
        raw_image = Image.open(img_path).convert('RGB')
        img = self.image_processor(raw_image)
        label = row['label']

        prompt_text = self.get_prompt(phrase)
        text = self.text_processor(prompt_text)
        return {
            'image': img,
            'text': text,
            'label': label,
        }

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def __len__(self):
        return len(self.dataset)


class IRFLDatasetEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
    def __init__(self,
                 dataset_dir,
                 model,
                 text_processor,
                 image_processor,
                 device,
                 batch_size,
                 shuffle_dataset,
                 output_file_path,
                 split,
                 ):
        super().__init__(dataset_dir)

        self.dataset_dir = dataset_dir
        self.model = model
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.device = device
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.output_file_path = output_file_path
        self.split = split
        self.prompt = IRFLPrompt()
        

    def evaluate_dataset(self,
                         metrics: List[HEMMMetric],
                         ) -> None:
        dataset = os.listdir(self.dataset_dir)
        pt_dataset = IRFLPTDataset(dataset, self.image_processor, self.text_processor, self.prompt, self.device)
        loader = torch.utils.data.DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        self.evaluate(self.model, loader, self.output_file_path, modalities=['img', 'text'])
