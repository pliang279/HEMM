# -*- coding: utf-8 -*-
"""VisualGenome_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pAPiREufnJssq8GkOG6DDsGU-tnJefK0
"""

import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.visualgenome_prompt import visualgenomeprompt

class VisualGenomeEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 questions_json_path,
                 device="cpu",
                 ):
        super().__init__()
        self.device = device
        self.prompt = visualgenomeprompt()
        self.questions_json_path = questions_json_path

        def get_prompt(self, text):
          prompt_text = self.prompt.format_prompt(text)
          return prompt_text

        def evaluate_dataset(self,
                         model,
                         metric
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric

        predictions = []
        ground_truth = []

        f = open(self.questions_json_path)


        data_vqa = json.load(f)
        for i in range(len(data_vqa)):
          temp_dict=data_vqa[i]
          img_id=temp_dict['id']
          qas=temp_dict['qas']
          try:
              if i==1:
                url=f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_id}.jpg"
                image=Image.open(requests.get(url, stream=True).raw)
                image_b = image.resize((640,480))
              else:
                url=f"https://cs.stanford.edu/people/rak248/VG_100K/{img_id}.jpg"
                image=Image.open(requests.get(url, stream=True).raw)
                image_b = image.resize((640,480))
          except:
              continue
          for j in range(len(qas)):
                question=qas[j]['question']
                question_pmt=self.get_prompt(question)
                output = self.model.generate(question_pmt, image_b)
                predictions.append(output)
                ground_truth.append(qas[j]['answer'])

        results = self.metric.compute(ground_truth, predictions)
        return results