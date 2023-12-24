import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.fer_prompt import FerPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class FERDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/.cache/test/',
                 annotation_file="/work/agoindan/.cache/test/test.txt",
                 device="cpu",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = FerPrompt()
        self.annotation_file = annotation_file
        self.metrics = [BertScoreMetric(), BleuMetric()]
    
    def __len__(self,):
        with open(self.annotation_file) as f:
            annotations = f.readlines()

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
        texts = []

        with open(self.annotation_file) as f:
            annotations = f.readlines()

        for row in tqdm(annotations, total=len(annotations)):            
            img_id = row.strip().split("./")[-1].split(".jpg")[0]
            label = img_id.split("/")[0]
            image_path = f"{self.image_dir}/{img_id}.jpg"
            # raw_images.append(Image.open(image_path).convert("RGB"))
            text = self.get_prompt()
            texts.append(text)
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
        raw_images = []

        with open(self.annotation_file) as f:
            annotations = f.readlines()

        for row in tqdm(annotations, total=len(annotations)):
            img_id = row.strip().split("./")[-1].split(".jpg")[0]
            label = img_id.split("/")[0]
            image_path = f"{self.image_dir}/{img_id}.jpg"
            raw_image = Image.open(image_path).convert('RGB')
            raw_images.append(raw_image)
            # image = self.model.get_image_tensor(raw_image)
            # images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(label)
        

        self.save_details(raw_images, texts, ground_truth, "fer.pkl")

        # predictions = self.predict_batched(images, texts, batch_size)

        return predictions, ground_truth
