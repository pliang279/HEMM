import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.enrico_prompt import EnricoPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class EnricoDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/enrico/screenshots',
                 annotation_file="/work/agoindan/enrico/design_topics.csv",
                 device="cpu",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = EnricoPrompt()
        self.annotation_file = annotation_file
        self.metrics = [BertScoreMetric(), BleuMetric()]
    
    def __len__(self,):
        with open(self.annotation_file) as f:
            annotations = f.readlines()
        annotations = annotations[1:]
        return len(annotations)

    def load(self):
        pass

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

        with open(self.annotation_file) as f:
            annotations = f.readlines()
        annotations = annotations[1:]

        for row in tqdm(annotations, total=len(annotations)):            
            img_id, label = row.strip().split(",")
            image_path = f"{self.image_dir}/{img_id}.jpg"
            text = self.get_prompt()
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

        with open(self.annotation_file) as f:
            annotations = f.readlines()
        annotations = annotations[1:]

        for row in tqdm(annotations, total=len(annotations)):
            img_id, label = row.strip().split(",")
            image_path = f"{self.image_dir}/{img_id}.jpg"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(label)

        samples = len(images) // 10
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth[:samples], predictions)

        return predictions, results, ground_truth[:samples]
