from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
from hemm.prompts.nlvr2prompt import NLVR2prompt
import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
from tqdm import tqdm
import random

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command

class NLVR2evaluator(HEMMDatasetEvaluator):
    def __init__(self, download_dir="./", dataset_dir=None, annotation_file="nlvr/nlvr2/data/dev.json", **kwargs):
        self.download_dir = download_dir
        annotation_file = os.path.join(download_dir, annotation_file)
        self.load()
        self.train_json = [json.loads(line) for line in open(annotation_file).readlines()]
        self.prompt = NLVR2prompt()

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def load(self):
        shell_command(f'git clone https://github.com/lil-lab/nlvr.git {self.download_dir}/nlvr/')

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        res = []
        gt=[]
        preds=[]
        texts = []
        for i in tqdm(range(len(self.train_json)), total=len(self.train_json)):
            left_url=self.train_json[i]['left_url']
            right_url=self.train_json[i]['right_url']
            label=self.train_json[i]['label']
            label=label.lower()
            sentence=self.train_json[i]['sentence']
            try:
              img_1=Image.open(requests.get(left_url, stream=True).raw)
              img_2=Image.open(requests.get(right_url, stream=True).raw)
              size1=img_1.size
              size2=img_2.size
              avg_size=((size1[0]+size2[0])//2,(size1[1]+size2[1])//2)
              img_1 =img_1.resize(avg_size)
              img_2=img_2.resize(avg_size)
              image = Image.new('RGB',(2*avg_size[0],avg_size[1]), (255,255,255))
              image.paste(img_1,(0,0))
              image.paste(img_2,(avg_size[0],0))
            except:
              continue
            text = self.get_prompt(sentence)
            texts.append(text)
            answer = model.generate(text, image)
            answer =''.join(filter(str.isalpha, answer.lower()))
            preds.append(answer)
            gt.append(label)

        return preds, gt
    
    def evaluate_dataset_batched(self, model=None, batch_size=None):
        pass
    