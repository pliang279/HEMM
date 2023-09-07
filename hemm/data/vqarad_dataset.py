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
        
        predictions = []
        ground_truth = []
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
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
        
        results = self.metric.compute(ground_truth, predictions)
        return results

    def evaluate_dataset_batched(self,
                         model,
                         metric,
                         batch_size=32
                         ):
        self.load()
        self.model = model
        self.metric = metric
        
        predictions = []
        ground_truth = []
        images = []
        texts = []
        for data_dict in tqdm(self.dataset, total=len(self.dataset)):
            question = data_dict['question']
            image = data_dict['image']
            with open("current_image.jpg", 'wb') as f:
                # f.write(image_url)
                image.save(f)
                image_path = "current_image.jpg"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            ground_truth_answer = data_dict['answer']
            text = self.get_prompt(question)
            texts.append(text)
            ground_truth.append(ground_truth_answer)
        
        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.chat.device)
        predictions = self.model.generate_batch(images_tensor, texts, batch_size)
        results = self.metric.compute(ground_truth, predictions)
        return results