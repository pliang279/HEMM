import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import random

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.resisc45_prompt import Resisc45Prompt

class Resisc45DatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.dataset_dir = download
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = Resisc45Prompt()
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.dataset_dir}/nwpu-data-set.zip'):
            shell_command(f'kaggle datasets download -d happyyang/nwpu-data-set -p {self.dataset_dir}')
        if not os.path.exists(f'{self.dataset_dict}/resisc45'):
            shell_command(f'unzip {self.dataset_dir}/nwpu-data-set.zip -d {self.dataset_dir}/resisc45')
        
        self.images_dir = f'{self.dataset_dir}/resisc45/NWPU Data Set/NWPU-RESISC45/NWPU-RESISC45'
        classes = os.listdir(self.images_dir)

        images_list = []
        ground_truth_list = []
        for image_class in classes:
            x = os.listdir(os.path.join(self.images_dir, image_class))
            images_list.extend(x)
            ground_truth_list.extend([image_class for i in range(len(x))])

        data_list = []
        for x, y in zip(images_list, ground_truth_list):
            data_list.append((x, y))

        random.shuffle(data_list)
        self.dataset = data_list

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None: 
        predictions = []
        ground_truth = []
        
        for data in tqdm(self.dataset, total=len(self.dataset)):
            image_path = os.path.join(self.images_dir, data[1], data[0])
            ground_truth_answer = data[1]
            text = self.get_prompt()
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
        
        texts = []
        images = []
        
        for data in tqdm(self.dataset, total=len(self.dataset)):
            image_path = os.path.join(self.images_dir, data[1], data[0])
            ground_truth_answer = data[1]
            text = self.get_prompt()
            texts.append(text)
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            ground_truth.append(ground_truth_answer)
        
        predictions = self.predict_batched(images, texts, batch_size)
        return predictions, ground_truth
    