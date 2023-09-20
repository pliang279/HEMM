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
from hemm.metrics.metric import HEMMMetric
from hemm.prompts.scienceqa_prompt import ScienceQAPrompt

class ScienceQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 ):
        super().__init__()
        self.dataset_key = 'scienceqa'
        self.prompt = ScienceQAPrompt()
    
    def get_prompt(self, question_s, choices, lecture, context) -> str:
        prompt_text = self.prompt.format_prompt(question_s, choices, lecture, context)
        return prompt_text

    def load(self):
        self.dataset = load_dataset("derek-thomas/ScienceQA")
        self.dataset = self.dataset['test']

    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        
        self.load()
        self.metric = metric
        self.model = model
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
            ground_truth.append(['answer'])
            question = self.get_prompt(question_s,
                                       choices,
                                       lecture,
                                       context
                                       )
            for ind, choice in enumerate(choices):
                curr_choice_str = str(ind+1) + ') ' + choice
                question = question + curr_choice_str + '\n'
            question += '\n'
            with open("current_image.jpg", 'wb') as f:
                # f.write(image_url)
                image_url.save(f)
                image_path = "current_image.jpg"
            
            ans = self.model.generate(question, image_path)
            predictions.append(ans)
        results = self.metric.compute(ground_truth, predictions)
        return results
    
    def evaluate_dataset_batched(self,
                         model,
                         metric,
                         batch_size=32
                         ) -> None:
        
        self.load()
        self.metric = metric
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
            ground_truth.append(['answer'])
            question = self.get_prompt(question_s,
                                       choices,
                                       lecture,
                                       context
                                       )
            for ind, choice in enumerate(choices):
                curr_choice_str = str(ind+1) + ') ' + choice
                question = question + curr_choice_str + '\n'
            question += '\n'
            texts.append(question)
            with open("current_image.jpg", 'wb') as f:
                # f.write(image_url)
                image_url.save(f)
                image_path = "current_image.jpg"
            
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)

        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.chat.device)
        outputs = self.model.generate_batch(images_tensor, texts, batch_size)
        results = self.metric.compute(ground_truth, outputs)
        return results