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
from hemm.prompts.scienceqa_prompt import ScienceQAPrompt

class ScienceQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self, download_dir="./", dataset_dir=None, annotation_file=None, **kwargs):
        super().__init__()
        self.download_dir = download_dir
        self.prompt = ScienceQAPrompt()
        self.load()
 
    def get_prompt(self, question_s, choices, lecture, context) -> str:
        prompt_text = self.prompt.format_prompt(question_s, choices, lecture, context)
        return prompt_text

    def load(self):
        self.dataset = load_dataset("derek-thomas/ScienceQA", cache_dir=self.download_dir)
        self.dataset = self.dataset['test']

    def __len__(self):
        return len(self.dataset)

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        
        predictions = []
        ground_truth = []
        for item in tqdm(self.dataset, total=len(self.dataset)):
            if not item['image']:
                continue
            question_s = item['question']
            choices = item['choices']
            lecture = item['lecture']
            image_url = item['image']
            context = item['hint']
            ground_truth.append(choices[item['answer']])
            question = self.get_prompt(lecture,
                                       question_s,
                                       context,
                                       choices
                                       )
            for ind, choice in enumerate(choices):
                curr_choice_str = str(ind+1) + ') ' + choice
                question = question + curr_choice_str + '\n'
            question += '\n'
            with open(f"{self.download_dir}/current_image.jpg", 'wb') as f:
                image_url.save(f)
                image_path = f"{self.download_dir}/current_image.jpg"
            
            ans = model.generate(question, image_path)
            predictions.append(ans)
        
        return predictions, ground_truth
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        
        self.model = model
        predictions = []
        ground_truth = []
        images = []
        texts = []
    
        for item in tqdm(self.dataset, total=len(self.dataset)):
            if not item['image']:
                continue
            question_s = item['question']
            choices = item['choices']
            lecture = item['lecture']
            image_url = item['image']
            context = item['hint']
            ground_truth.append(choices[item['answer']])
            question = self.get_prompt(lecture,
                                       question_s,
                                       context,
                                       choices
                                       )
            for ind, choice in enumerate(choices):
                curr_choice_str = str(ind+1) + ') ' + choice
                question = question + curr_choice_str + '\n'
            question += '\n'
            texts.append(question)
            with open(f"{self.download_dir}/current_image.jpg", 'wb') as f:
                image_url.save(f)
                image_path = f"{self.download_dir}/current_image.jpg"
            
            raw_image = Image.open(image_path).convert('RGB')

            image = self.model.get_image_tensor(raw_image)
            images.append(image)

        predictions = self.predict_batched(images, texts, batch_size)        
        return predictions, ground_truth
    