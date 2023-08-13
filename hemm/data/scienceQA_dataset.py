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
            number = self.model.answer_extractor(ans, self.dataset_key)
            if number:
                predictions.append(number)
            else:
                random_item = random.choice(list(range(0, len(choices))))
            predictions.append(random_item)
        
        results = self.metric.compute(ground_truth, predictions)
        return results