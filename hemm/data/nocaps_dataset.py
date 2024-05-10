import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import subprocess
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.nocaps_prompt import NoCapsPrompt
from hemm.utils.common_utils import shell_command
from huggingface_hub import snapshot_download

class NoCapsDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                    download_dir="./", 
                    dataset_dir="nocaps_images/",
                    annotation_file='nocaps_images/nocaps_val_4500_captions.json',
                    **kwargs):
        super().__init__()
        self.annotation_file = os.path.join(download_dir, annotation_file)
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = NoCapsPrompt()
        self.load()

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self):
        json_file = json.load(open(self.annotation_file, 'r'))
        return len(json_file["images"])

    def load(self):
        if not os.path.exists(self.dataset_dir):
            shell_command(f"mkdir -p {self.dataset_dir}")
            snapshot_download(repo_id="akshayg08/NocapsTest", repo_type="dataset", local_dir=self.dataset_dir)
            shell_command(f"wget https://s3.amazonaws.com/nocaps/nocaps_val_4500_captions.json -P {self.dataset_dir}")
    
    def evaluate_dataset(self,
                         model,
                         ) -> None:

        json_file = json.load(open(self.annotation_file, 'r'))
        predictions = []
        ground_truth = []
        for index, image_dict in tqdm(enumerate(json_file['images']), total=len(json_file['images'])):
            fn = image_dict["file_name"]
            captions = []
            for ann in json_file["annotations"][10*index: 10*(index + 1)]:
                captions.append(ann["caption"])
            
            ground_truth.append(captions)
            text = self.get_prompt()

            image_path = os.path.join(self.dataset_dir, fn)
            output = model.generate(text, image_path)
            predictions.append(output)

        return predictions, ground_truth
 
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):
        self.model = model
        json_file = json.load(open(self.annotation_file, 'r'))
        predictions = []
        ground_truth = []

        texts = []
        images = []

        for index, image_dict in tqdm(enumerate(json_file['images']), total=len(json_file['images'])):
            fn = image_dict["file_name"]

            captions = []
            for ann in json_file["annotations"][10*index: 10*(index + 1)]:
                captions.append(ann["caption"])
            
            ground_truth.append(captions)
            text = self.get_prompt()
            image_path = os.path.join(self.dataset_dir, fn)
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)            
            texts.append(text)
        
        predictions = self.predict_batched(images, texts, batch_size)
        return predictions, ground_truth
    