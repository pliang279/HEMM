import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.magic_brush_prompt import MagicBrushPrompt
from hemm.metrics.image_match_metric import MSEMetric, CLIPIMetric

class MagicBrushDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir="/work/agoindan/magic_brush/test/images",
                 annotation_file="/work/agoindan/magic_brush/test/edit_sessions.json",
                 device="cpu",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = MagicBrushPrompt()
        self.annotation_file = annotation_file

        # TODO
        self.metrics = [MSEMetric(), CLIPIMetric(self.device)]

    def load(self):
        pass

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

        annotations = json.load(open(self.annotation_file))
        idx = 0
        for img_id in annotations:
            ann = annotations[img_id]
            for sample in ann:
                if idx == 10:
                    break
                input_img = f"{self.image_dir}/{img_id}/{sample['input']}"
                text = self.get_prompt(sample['instruction'])
                gt_img = f"{self.image_dir}/{img_id}/{sample['output']}"
                pred_img = self.model.generate_image(text, input_img)
                predictions.append(gt_img)
                ground_truth.append(pred_img)
                idx += 1
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results
    
    def evaluate_dataset_batched(self):
        pass
    
    # def evaluate_dataset_batched(self,
    #                      model,
    #                      metric,
    #                      batch_size=32
    #                      ) -> None:
    #     self.load()
    #     self.model = model
    #     self.metric = metric
        
    #     predictions = []
    #     ground_truth = []

    #     texts = []
    #     images = []

    #     with open(self.annotation_file) as f:
    #         annotations = f.readlines()
    #     annotations = annotations[1:]

    #     for row in tqdm(annotations, total=len(annotations)):
    #         img_id, label = row.strip().split(",")
    #         image_path = f"{self.image_dir}/{img_id}.jpg"
    #         raw_image = Image.open(image_path).convert('RGB')
    #         image = self.model.get_image_tensor(raw_image)
    #         images.append(image)
    #         text = self.get_prompt()
    #         texts.append(text)
    #         ground_truth.append(label)
        
    #     images_tensor = torch.cat(images, dim=0)
    #     images_tensor = images_tensor.to(self.model.chat.device)
    #     predictions = self.model.generate_batch(images_tensor, texts, batch_size)

    #     results = self.metric.compute(ground_truth, predictions)
    #     return results
        