import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import random
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
        self.metrics = [BertScoreMetric(), BleuMetric()]
        random.seed(0)
        with open(annotation_file) as f:
            self.annotations = f.readlines()
        
        self.annotations = self.annotations[1:]
        random.shuffle(self.annotations)
        self.annotations = self.annotations[-len(self.annotations) // 10:]
    
    def __len__(self,):
        return len(self.annotations)

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

        for row in tqdm(self.annotations, total=len(self.annotations)):            
            img_id, label = row.strip().split(",")
            image_path = f"{self.image_dir}/{img_id}.jpg"
            # images.append(image_path)
            text = self.get_prompt()
            # texts.append(text)
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)
        
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

        for row in tqdm(self.annotations, total=len(self.annotations)):
            img_id, label = row.strip().split(",")
            image_path = f"{self.image_dir}/{img_id}.jpg"
            raw_image = Image.open(image_path).convert('RGB')
            # raw_images.append(raw_image)
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(label)
        
        samples = len(images)
        # self.save_details(raw_images[:samples], texts[:samples], ground_truth[:samples], "enrico.pkl")
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)

        return predictions, ground_truth[:samples]
