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

    def __len__(self):
        annotations = json.load(open(self.annotation_file))
        return len(annotations)

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
        samples = len(annotations) // 10
        annotations = annotations[:samples]

        for img_id in tqdm(annotations, total=len(annotations)):
            ann = annotations[img_id]
            for sample in ann:
                input_img = f"{self.image_dir}/{img_id}/{sample['input']}"
                text = self.get_prompt(sample['instruction'])
                gt_img = f"{self.image_dir}/{img_id}/{sample['output']}"
                pred_img = self.model.generate_image(text, input_img)
                predictions.append(pred_img)
                ground_truth.append(gt_img)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results, ground_truth
    
    def evaluate_dataset_batched(self):
        pass
        