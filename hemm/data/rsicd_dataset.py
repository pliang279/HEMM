import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import subprocess
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.prompts.rsicd_prompt import RSICDPrompt
from hemm.utils.common_utils import shell_command

class RSICDDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 ):
        self.prompt = RSICDPrompt()

    def get_prompt(self,) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def load(self):
        shell_command('apt install megatools')
        if not os.path.exists('RSICD_images.rar'):
          shell_command('megadl https://mega.nz/folder/EOpjTAwL#LWdHVjKAJbd3NbLsCvzDGA')
        if not os.path.exists('rsicd_images'):
          shell_command('mkdir rsicd_images')
          shell_command('!unrar e /content/RSICD_images.rar rsicd_images')
        shell_command('git clone https://github.com/201528014227051/RSICD_optimal.git')
        
    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        self.load()
        self.metric = metric
        self.model = model

        images_dir = 'rsicd_images'
        annotation_path = 'RSICD_optimal/dataset_rsicd.json'
        annotation_file = json.load(open(annotation_path))
        
        predictions = []
        ground_truth = []
        for index, data_dict in tqdm(enumerate(annotation_file['images']), total=len(annotation_file['images'])):
            image_path = os.path.join(images_dir, data_dict['filename'])
            gt_caption = data_dict["sentences"][0]['raw']
            text = self.get_prompt()
            ground_truth.append(gt_caption)
            output = self.model.generate(text, image_path)
            predictions.append(output)
        
        results = self.metric.compute(ground_truth, predictions)
        return results