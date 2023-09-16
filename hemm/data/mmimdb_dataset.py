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

class MMIMDBDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/mmimdb/dataset',
                 annotation_dir="/work/agoindan/mmimdb/dataset",
                 device="cpu",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = MMIMDBPrompt()
        self.annotation_dir = annotation_dir

    # def load(self):
    #   os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
    #   if not os.path.exists('landuse-scene-classification.zip'):
    #       shell_command('kaggle datasets download -d apollo2506/landuse-scene-classification')
    #   if not os.path.exists('ucmercedimages'):
    #       shell_command('unzip landuse-scene-classification.zip -d ucmercedimages/')

    def load(self):
        pass

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
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

        ann_files = glob(f"{self.annotation_dir}/*.json")

        for row in tqdm(ann_files, total=len(ann_files)):
            ann_id = row.strip().split(".json")[0].split("/")[-1]
            image_path = f"{self.image_dir}/{ann_id}.jpeg"
            data = json.load(open(row))
            text = self.get_prompt(data["plot"][0])
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)
        
        results = self.metric.compute(ground_truth, predictions)
        return results
    
    def evaluate_dataset_batched(self,
                         model,
                         metric,
                         batch_size=32
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric
        
        predictions = []
        ground_truth = []

        texts = []
        images = []

        ann_files = glob(f"{self.annotation_dir}/*.json")

        for row in tqdm(ann_files, total=len(ann_files)):
            ann_id = row.strip().split(".json")[0].split("/")[-1]
            image_path = f"{self.image_dir}/{ann_id}.jpeg"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            data = json.load(open(row))
            text = self.get_prompt(data["plot"][0])
            texts.append(text)
            ground_truth.append(label)
        
        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.chat.device)
        predictions = self.model.generate_batch(images_tensor, texts, batch_size)

        results = self.metric.compute(ground_truth, predictions)
        return results
        