import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import json

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.slake_prompt import SlakePrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class SlakeDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/Slake1.0/imgs',
                 annotation_file="/work/agoindan/Slake1.0/test.json",
                 device="cpu",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = SlakePrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]
        all_annotation = json.load(open(annotation_file))
        self.annotation = []
        for ann in all_annotation:
            if ann["q_lang"] == "en" and ann["answer_type"] == "CLOSED":
                self.annotation.append(ann)

    def load(self):
        pass

    def __len__(self):
        return len(self.annotation)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []

        for row in tqdm(self.annotation, total=len(self.annotation)):
            label = row["answer"]
            question = row["question"]
            image_path = f"{self.image_dir}/{row['img_name']}"
            text = self.get_prompt(question)
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)

        return predictions, results
    
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
            label = row["answer"]
            question = row["question"]
            image_path = f"{self.image_dir}/{row['img_name']}"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt(question)
            texts.append(text)
            ground_truth.append(label)
        
        samples = len(images) // 10
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth[:samples], predictions)

        return predictions, results, ground_truth[:samples]
