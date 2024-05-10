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

class IRFLDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self, download_dir="./", dataset_dir=None, annotation_file=None, **kwargs):
        super().__init__()
        self.download_dir = download_dir
        self.prompt = IRFLPrompt()
        self.load()

    def load(self):
        self.IRFL_images = load_dataset("lampent/IRFL", data_files='IRFL_images.zip', cache_dir=self.download_dir)['train']
        self.dataset = load_dataset("lampent/IRFL", "simile-detection-task", cache_dir=self.download_dir)["test"]
        self.dataset = pd.DataFrame(self.dataset)

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text    

    def evaluate_dataset(self,
                         model,
                         ) -> None:

        self.model = model
        predictions = []
        outputs = []
        ground_truth = []
        texts = []
        for index, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
            phrase = row['phrase']
            distractors = ast.literal_eval(row['distractors'])
            question = self.get_prompt(phrase)
            answer_image = ast.literal_eval(row['answer'])[0]
            distractors.append(answer_image)
            
            for distractor in distractors:
                image_path = self.get_image_path_from_hugginface_cache(distractor)
                texts.append(question)
                answer = self.model.generate(question, image_path)
                outputs.append(answer)

            ground_truth += ["no", "no", "no", "yes"]
        return outputs, ground_truth

    def get_image_path_from_hugginface_cache(self, image_name):
        chached_image_path = self.IRFL_images[0]['image'].filename
        chached_image_name = chached_image_path.split('/')[-1]
        return chached_image_path.replace(chached_image_name, image_name.split('.')[0] + '.jpeg')
        
    def evaluate_dataset_batched(self,
                         model=None,
                         batch_size=32
                         ):
      pass
