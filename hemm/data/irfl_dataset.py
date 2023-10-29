import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import pandas as pd
import ast
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.irfl_prompt import IRFLPrompt
from hemm.utils.common_utils import shell_command
from hemm.metrics.accuracy_metric import *

class IRFLDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 csv_path = 'IRFL/assets/tasks/simile_understanding_task.csv',
                 ):
        super().__init__()
        self.dataset_key = 'irfl'
        self.csv_path = csv_path
        self.prompt = IRFLPrompt()
        self.metrics = [AccuracyMetric(), PrecisionMetric(), RecallMetric(), F1ScoreMetric()]

    def load(self):
        shell_command('git clone https://github.com/irfl-dataset/IRFL')

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text    

    def evaluate_dataset(self,
                         model,
                         ) -> None:

        self.load()
        self.model = model
        df = pd.read_csv(self.csv_path)
        predictions = []
        outputs = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                phrase = row['phrase']
                distractors = ast.literal_eval(row['distractors'])
                question = self.get_prompt(phrase)
                answer_image = ast.literal_eval(row['answer'])[0]
                generated_answers = []
                distractors.append(answer_image)
                for distractor in distractors:
                    response = requests.get(distractor)
                    if response.status_code == 200:
                        with open("current_image.jpg", 'wb') as f:
                            f.write(response.content)
                    image_path = "current_image.jpg"
                    answer = self.model.generate(question, image_path)
                    outputs.append(answer)
                    answer = self.model.answer_extractor(answer, self.dataset_key)
                    generated_answers.append(answer)

                if generated_answers[-1] == 'yes':
                    if generated_answers[0] == 'no' and generated_answers[1] == 'no' and generated_answers[2] == 'no':
                        predictions.append(1)
                    else:
                        predictions.append(0)
                else:
                    predictions.append(0)
            except Exception as e:
                continue

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return outputs, results
        
    def evaluate_dataset_batched(self,
                         metric,
                         model,
                         batch_size=32
                         ):
      pass
