import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
import pandas as pd
import subprocess
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator

from hemm.prompts.newyorkercartoon_prompt import NewYorkerCartoonPrompt
from hemm.utils.common_utils import shell_command

class NewYorkerCartoonDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='caption-contest-data/',
                annotation_file=None,
                **kwargs, 
                ):
        super().__init__()
        self.download_dir = download_dir
        self.dataset_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = NewYorkerCartoonPrompt()

        self.image_dir = os.path.join(self.dataset_dir, 'cartoons')
        self.caption_dir = os.path.join(self.dataset_dir, 'summaries')
        self.csv_path_suffix_1 = 'LilUCB'
        self.csv_path_suffix_2 = 'lil-KLUCB'

        self.load()

    def load(self):
        if not os.path.exists(f"{self.download_dir}/caption-contest-data"):
            shell_command(f'git clone https://github.com/nextml/caption-contest-data {os.path.join(self.download_dir, "caption-contest-data")}')

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text
    
    def __len__(self):
        return len(os.listdir(self.image_dir))

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        
        ground_truth = []
        outputs = []
        
        for img in tqdm(os.listdir(self.image_dir), total=len(os.listdir(self.image_dir))):
            img_id = img.split('.jpg')[0]
            img_path = os.path.join(self.image_dir, img)
            if os.path.exists(os.path.join(self.caption_dir, img_id+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+ '.csv'))
            elif os.path.exists(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_1+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_1+'.csv'))
            elif os.path.exists(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_2+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_2+'.csv'))
            
            captions = []
            captions.append(df.iloc[0]['caption'])

            for i in range(1, 5):
                captions.append(df.iloc[-1*i]['caption'])
            
            for i in range(len(captions)):
                text = self.get_prompt(captions[i])
                
                output = model.generate(text, img_path)
                
                outputs.append(output)

                if i == 0:
                    ground_truth.append("yes")
                else:
                    ground_truth.append("no")
        
        return outputs, ground_truth
    
    def evaluate_dataset_batched(self,
                                model,
                                batch_size=32
                                ):
        
        self.model = model
        texts = []
        images = []
        ground_truth_list = []
        predictions = []

        for img in tqdm(os.listdir(self.image_dir), total=len(os.listdir(self.image_dir))):
            img_id = img.split('.jpg')[0]
            img_path = os.path.join(self.image_dir, img)
            if os.path.exists(os.path.join(self.caption_dir, img_id+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+ '.csv'))
            elif os.path.exists(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_1+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_1+'.csv'))
            elif os.path.exists(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_2+'.csv')):
                df = pd.read_csv(os.path.join(self.caption_dir, img_id+"_"+self.csv_path_suffix_2+'.csv'))
            
            captions = []
            captions.append(df.iloc[0]['caption'])

            for i in range(1, 5):
                captions.append(df.iloc[-1*i]['caption'])
            
            for i in range(len(captions)):
                text = self.get_prompt(captions[i])
                texts.append(text)
                raw_image = Image.open(img_path).convert('RGB')
                image = self.model.get_image_tensor(raw_image)
                images.append(image)
                if i == 0:
                    ground_truth_list.append("yes")
                else:
                    ground_truth_list.append("no")
        
        outputs = self.predict_batched(images, texts, batch_size)
        return outputs, ground_truth_list    

