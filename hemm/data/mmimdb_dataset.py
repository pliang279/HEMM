import os
import json
from glob import glob
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from hemm.data.dataset import HEMMDatasetEvaluator

from hemm.utils.common_utils import shell_command
from hemm.prompts.mmimdb_prompt import MMIMDBPrompt
from huggingface_hub import snapshot_download

class MMIMDBDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='mmimdb/',
                annotation_file="mmimdb/split.json",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        self.prompt = MMIMDBPrompt()
        self.annotation_file = os.path.join(download_dir, annotation_file)
        self.load()

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f"{self.download_dir}/mmimdb"):
            shell_command(f"mkdir -p {self.download_dir}/mmimdb")
            snapshot_download(repo_id="akshayg08/mmimdb_test", repo_type="dataset", local_dir=f"{self.download_dir}/mmimdb/")
            
    def __len__(self):
        ann_files = json.load(open(self.annotation_file))["test"]
        return len(ann_files)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None: 
        predictions = []
        ground_truth = []

        ann_files = json.load(open(self.annotation_file))["test"]
        
        for row in tqdm(ann_files, total=len(ann_files)):
            ann_id = row.strip()
            image_path = f"{self.image_dir}/{ann_id}.jpeg"
            data = json.load(open(f"{self.dataset_dir}/annotations/{ann_id}.json"))
            text = self.get_prompt(data["plot"][0])
            label = ", ".join(data["genres"])
            
            output = model.generate(text, image_path)
            
            predictions.append(output)
            ground_truth.append(label)
            
        return predictions, ground_truth
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.model = model
        
        ground_truth = []

        texts = []
        images = []


        all_results = []
        ann_files = json.load(open(self.annotation_file))["test"]
        for row in tqdm(ann_files, total=len(ann_files)):
            ann_id = row.strip()
            image_path = f"{self.image_dir}/{ann_id}.jpeg"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            data = json.load(open(f"{self.dataset_dir}/annotations/{ann_id}.json"))
            text = self.get_prompt(data["plot"][0])
            label = ", ".join(data["genres"])
            texts.append(text)
            ground_truth.append(label.strip())

            if len(images) % batch_size == 0:
                all_results += self.predict_batched(images, texts, batch_size)
                images = []
                texts = []

        if len(images) > 0:
            all_results += self.predict_batched(images, texts, batch_size)

        return all_results, ground_truth
        