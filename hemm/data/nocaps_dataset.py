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
from hemm.prompts.nocaps_prompt import NoCapsPrompt


class NoCapsPTDataset(torch.utils.data.Dataset):
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
        image_dict = self.dataset['images'][index]
        image_url = image_dict['coco_url']
        image_caption = self.dataset['annotations'][image_dict['id']]['caption']
        img = self.image_processor(image_url)
        caption_label = self.text_processor(image_caption)

        prompt_text = self.get_prompt()
        text = self.text_processor(prompt_text)
        return {
            'image': img,
            'text': text,
            'caption_label': caption_label,
        }

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def __len__(self):
        return len(self.dataset)


class NoCapsDatasetEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
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
        """
        Class for hateful memes dataset. Assuming data is already downloaded at dataset_path directory.
        """
        self.dataset_dir = dataset_dir
        self.model = model
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.device = device
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.output_file_path = output_file_path
        self.split = split
        self.prompt = NoCapsPrompt()

    def evaluate_dataset(self,
                         metrics: List[HEMMMetric],
                         ) -> None:
        json_file = json.load(open(self.dataset_dir, 'r'))
        pt_dataset = NoCapsPTDataset(json_file, self.image_processor, self.text_processor, self.prompt,
                                     self.device)
        loader = torch.utils.data.DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        self.evaluate(self.model, loader, self.output_file_path, modalities=['img', 'text'])
