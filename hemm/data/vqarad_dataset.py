import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.vqarad_prompt import VQARADPrompt


class VQARADDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./'
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.prompt = VQARADPrompt()

    def load(self):
      self.dataset = load_dataset("flaviagiammarino/vqa-rad")
      self.dataset = self.dataset['test']

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         metric
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric
        
        acc = []
        for data_dict in tqdm(self.dataset, total=len(self.dataset)):
            question = data_dict['question']
            image = data_dict['image']
            with open("current_image.jpg", 'wb') as f:
                # f.write(image_url)
                image.save(f)
                image_path = "current_image.jpg"
            ground_truth_answer = data_dict['answer']
            text = self.get_prompt(question)
            output = self.model.generate(text, image_path)
            if output == ground_truth_answer:
                acc.append(1)
            else:
                acc.append(0)
        
        return sum(acc) / len(acc)