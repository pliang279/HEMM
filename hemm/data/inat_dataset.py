import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import subprocess
from glob import glob

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.inat_prompt import INATPrompt
from hemm.utils.common_utils import shell_command
from hemm.metrics.accuracy_metric import *
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class INATDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/inat/val',
                 kaggle_api_path = None,
                 ):
        super().__init__()

        self.image_dir = image_dir
        self.kaggle_api_path = kaggle_api_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.kaggle_api_path = kaggle_api_path
        self.prompt = INATPrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]
    
    def load(self, kaggle_api_path):
        pass

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
    
        self.model = model
        all_images = glob(f"{self.image_dir}/*/*.jpg")

        ground_truth = []
        predictions = []

        for idx in tqdm(range(len(all_images)), total=len(all_images)):
            if idx == 100:
                break
            image_path = all_images[idx]
            text = self.get_prompt()
            output = self.model.generate(text, image_path)

            gt_name = " ".join(all_images[idx].split("/")[-2].split("_")[-2:])
            ground_truth.append(gt_name.lower())
            predictions.append(output)

        results = {}
        for metric in self.metrics:
            metric_val = metric.compute(ground_truth, predictions)
            results[metric.name] = metric_val
        
        return results
     
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.model = model
        all_images = glob(f"{self.image_dir}/*/*.jpg")

        ground_truth = []
        predictions = []

        images = []
        texts = []

        for idx in tqdm(range(len(all_images)), total=len(all_images)):
            if idx == 100:
                break
            image_path = all_images[idx]
            text = self.get_prompt()

            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            texts.append(text)

            gt_name = " ".join(all_images[idx].split("/")[-2].split("_")[-2:])
            ground_truth.append(gt_name)

        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.device)

        predictions = self.model.generate_batch(images_tensor, texts, batch_size)
        # for output in outputs:
        #     answer = self.model.answer_extractor(output, self.dataset_key)
        #     if answer == 'yes':
        #         predictions.append(1)
        #     else:
        #         predictions.append(0)
        
        results = {}
        for metric in self.metrics:
            metric_val = metric.compute(ground_truth, predictions)
            results[metric.name] = metric_val

        return results
    