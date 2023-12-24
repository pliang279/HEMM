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
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class ScienceQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 ):
        super().__init__()
        self.dataset_key = 'scienceqa'
        self.prompt = ScienceQAPrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load()
 
    def get_prompt(self, question_s, choices, lecture, context) -> str:
        prompt_text = self.prompt.format_prompt(question_s, choices, lecture, context)
        return prompt_text

    def load(self):
        self.dataset = load_dataset("derek-thomas/ScienceQA")
        self.dataset = self.dataset['test']

    def __len__(self):
        return len(self.dataset)

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        
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
            with open("current_image.jpg", 'wb') as f:
                image_url.save(f)
                image_path = "current_image.jpg"
            
            ans = self.model.generate(question, image_path)
            predictions.append(ans)
        
        return predictions, ground_truth
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        
        self.load()
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
            with open("current_image.jpg", 'wb') as f:
                image_url.save(f)
                image_path = "current_image.jpg"
            
            raw_image = Image.open(image_path).convert('RGB')

            image = self.model.get_image_tensor(raw_image)
            images.append(image)

        samples = len(images)
        # print(samples)
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)
        # print(len(raw_images))
        # samples = len(raw_images)
        # self.save_details(raw_images[:samples], texts[:samples], ground_truth[:samples], "scienceqa.pkl")
        
        return predictions, ground_truth[:samples]
    