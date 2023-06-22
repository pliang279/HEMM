import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import pandas as pd

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.utils.evaluator_mixin import EvaluatorMixin
from hemm.prompts.newyorkercartoon_prompt import NewYorkerCartoonPrompt


class NewYorkerCartoonPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 image_processor,
                 text_processor,
                 prompt,
                 image_dir,
                 caption_dir,
                 csv_path_suffix_1,
                 csv_path_suffix_2,
                 device,
                 ):
        self.dataset = dataset
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.device = device
        self.prompt = prompt

        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.csv_path_suffix_1 = csv_path_suffix_1
        self.csv_path_suffix_2 = csv_path_suffix_2

    def __getitem__(self, index):
        img = self.dataset[index]
        img_id = img.split('.jpg')[0]
        img_path = os.path.join(self.image_dir, img)
        if os.path.exists(os.path.join(self.caption_dir, img_id + '.csv')):
            df = pd.read_csv(os.path.join(self.caption_dir, img_id + '.csv'))
        elif os.path.exists(os.path.join(self.caption_dir, img_id + "_" + self.csv_path_suffix_1 + '.csv')):
            df = pd.read_csv(os.path.join(self.caption_dir, img_id + "_" + self.csv_path_suffix_1 + '.csv'))
        elif os.path.exists(os.path.join(self.caption_dir, img_id + "_" + self.csv_path_suffix_1 + '.csv')):
            df = pd.read_csv(os.path.join(self.caption_dir, img_id + "_" + self.csv_path_suffix_1 + '.csv'))

        caption = df.iloc[0]['caption']
        raw_image = Image.open(img_path).convert('RGB')
        img = self.image_processor(raw_image)
        caption_label = self.text_processor(caption)

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


class NewYorkerCartoonDatasetEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
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
        self.prompt = NewYorkerCartoonPrompt()

        self.image_dir = os.path.join(self.dataset_dir, 'cartoons')
        self.caption_dir = os.path.join(self.dataset_dir, 'summaries')
        self.csv_path_suffix_1 = 'LilUCB'
        self.csv_path_suffix_2 = 'lil-KLUCB'

    def evaluate_dataset(self,
                         metrics: List[HEMMMetric],
                         ) -> None:
        image_files = os.listdir(self.image_dir)
        pt_dataset = NewYorkerCartoonPTDataset(image_files, self.image_processor, self.text_processor, self.prompt,
                                               self.image_dir, self.caption_dir, self.csv_path_suffix_1, self.csv_path_suffix_2,
                                               self.device)
        loader = torch.utils.data.DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        self.evaluate(self.model, loader, self.output_file_path, modalities=['img', 'text'])
