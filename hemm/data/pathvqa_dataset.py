import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import subprocess
from tqdm import tqdm
import pickle

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.prompts.pathvqa_prompt import PathVQAPrompt
from hemm.utils.common_utils import shell_command

class PathVQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 ):
        self.prompt = PathVQAPrompt()

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def load(self):
        if not os.path.exists('Backup'):
          shell_command('gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1G2C2_FUCyYQKCkSeCRRiTTsLDvOAjFj5')
        if not os.path.exists('pathvqa_images'):
          shell_command('unzip Backup/pvqa.zip -d pathvqa_images/')
        
    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        self.load()
        self.metric = metric
        self.model = model
        
        images_dir = os.path.join('pathvqa_images','pvqa','images','test')
        annotation_path = os.path.join('pathvqa_images','pvqa','qas','test','test_qa.pkl')
        annotation_file = pickle.load(open(annotation_path, 'rb'))
        
        ground_truth = []
        predictions = []
        for index, data_dict in tqdm(enumerate(annotation_file), total=len(annotation_file)):
            image_path = os.path.join(images_dir, data_dict['image'] + '.jpg')
            question = data_dict['question']
            ground_truth_answer = data_dict["answer"]
            text = self.get_prompt(question)
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
        
        results = self.metric.compute(ground_truth, predictions)
        return results