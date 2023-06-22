import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.utils.evaluator_mixin import EvaluatorMixin
from hemm.prompts.scienceqa_prompt import ScienceQAPrompt


class ScienceQAPTDataset(torch.utils.data.Dataset):
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
        question_s = self.dataset[index]['question']
        choices = self.dataset[index]['choices']
        lecture = self.dataset[index]['lecture']
        image_url = self.dataset[index]['image']
        context = self.dataset[index]['hint']
        choice_label = choices[self.dataset[index]['answer']]
        solution_label = self.dataset[index]['solution']

        img = self.image_processor(image_url)
        choice_label = self.text_processor(choice_label)
        solution_label = self.text_processor(solution_label)

        prompt_text = self.get_prompt(question_s, choices, lecture, context)
        text = self.text_processor(prompt_text)
        return {
            'image': img,
            'text': text,
            'choice_label': choice_label,
            'solution_label': solution_label,
        }

    def get_prompt(self, question_s, choices, lecture, context) -> str:
        prompt_text = self.prompt.format_prompt(question_s, choices, lecture, context)
        return prompt_text

    def __len__(self):
        return len(self.dataset)


class ScienceQADatasetEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
    def __init__(self,
                 dataset_dir,
                 model,
                 text_processor,
                 image_processor,
                 device,
                 batch_size,
                 shuffle_dataset,
                 output_file_path,
                 split
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
        self.prompt = ScienceQAPrompt()

    def evaluate_dataset(self,
                         metrics: List[HEMMMetric],
                         ) -> None:
        dataset = load_dataset(self.dataset_dir)
        pt_dataset = ScienceQAPTDataset(dataset[self.split], self.image_processor, self.text_processor, self.prompt,
                                        self.device)
        loader = torch.utils.data.DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        self.evaluate(self.model, loader, self.output_file_path, modalities=['img', 'text'])
