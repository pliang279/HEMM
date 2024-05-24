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
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download_dir
        self.prompt = VQARADPrompt()
        self.load()

    def load(self):
        self.dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir=self.dataset_dir)
        self.dataset = self.dataset['test']

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
        
        for data_dict in tqdm(self.dataset, total=len(self.dataset)):
            question = data_dict['question']
            image = data_dict['image']
            with open(f"{self.dataset_dir}current_image.jpg", 'wb') as f:
                image.save(f)
                image_path = f"{self.dataset_dir}current_image.jpg"
            ground_truth_answer = data_dict['answer']
            text = self.get_prompt(question)
            
            output = model.generate(text, image_path)
            
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
            
        return predictions, ground_truth

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):
        self.model = model
        
        predictions = []
        ground_truth = []
        images = []
        texts = []
        
        for data_dict in tqdm(self.dataset, total=len(self.dataset)):
            question = data_dict['question']
            image = data_dict['image']
            with open(f"{self.dataset_dir}current_image.jpg", 'wb') as f:
                image.save(f)
                image_path = f"{self.dataset_dir}current_image.jpg"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            ground_truth_answer = data_dict['answer']
            text = self.get_prompt(question)
            texts.append(text)
            ground_truth.append(ground_truth_answer)
        
        predictions = self.predict_batched(images, texts, batch_size)

        return predictions, ground_truth
        