import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.plip_kather_prompt import PlipKatherPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class PlipKatherDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/CRC-VAL-HE-7K',
                 annotation_file="/work/agoindan/External_validation_data/Kather_test/Kather_test.csv",
                 device="cuda",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = PlipKatherPrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]
        with open(annotation_file) as f:
            self.annotation = f.readlines()
        self.annotation = self.annotation[1:]

    def load(self):
        pass

    def __len__(self):
        return len(self.annotation)

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []
        cnt = 0
        for row in tqdm(self.annotation, total=len(self.annotation)):
            _, fn, lb, caption = row.strip().split(",")
            label = " ".join(caption.split()[5:])[:-1]
            image_path = f"{self.image_dir}/{lb}/{fn}"
            text = self.get_prompt()
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)
            cnt += 1

        return predictions, ground_truth
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []

        texts = []
        images = []

        for row in tqdm(self.annotation, total=len(self.annotation)):
            _, fn, lb, caption = row.strip().split(",")
            label = " ".join(caption.split()[5:])[:-1]
            image_path = f"{self.image_dir}/{lb}/{fn}"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(label)
        
        samples = len(images)
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)
        # samples = len(raw_images)
        # self.save_details(raw_images[:samples], texts[:samples], ground_truth[:samples], "plip_kather.pkl")

        return predictions, ground_truth[:samples]

