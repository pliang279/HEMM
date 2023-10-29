import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.pmcvqa_prompt import PMCVQAPrompt
from hemm.metrics.accuracy_metric import *

class PMCVQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./'
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.prompt = PMCVQAPrompt()
        self.metrics = [AccuracyMetric(), PrecisionMetric(), RecallMetric(), F1ScoreMetric()]

    def load(self):
      if not os.path.exists('images.zip'):
        shell_command('wget https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/images.zip')
      if not os.path.exists('pmcvqa'):
        shell_command('unzip images.zip -d pmcvqa/')
      if not os.path.exists('test_clean.csv'):
        shell_command('wget https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/test_clean.csv')
      

    def get_prompt(self, question, choice_a, choice_b, choice_c, choice_d):
        prompt_text = self.prompt.format_prompt(question, choice_a, choice_b, choice_c, choice_d)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        
        self.annotation_file = 'test_clean.csv'
        self.image_dir = 'pmvcqa'

        self.dataset = pd.read_csv(self.annotation_file)

        ground_truth = []
        predictions = []
        for index, row in self.dataset.iterrows():
            question = row['Question']
            image_path = os.path.join(self.image_dir, 'images', row['Figure_path'])
            ground_truth_answer = row['Answer_label']
            choice_a, choice_b, choice_c, choice_d = row['Choice A'], row['Choice B'], row['Choice C'], row['Choice D']
            text = self.get_prompt(question, choice_a, choice_b, choice_c, choice_d)
            output = self.model.generate(text, image_path)
            ground_truth.append(ground_truth_answer)
            predictions.append(output)

        results = {}
        for metric in self.metrics:
          results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results
        