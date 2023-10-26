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
from hemm.metrics.accuracy_metric import * 

class NewYorkerCartoonDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir = './caption-contest-data/',
                 ):
        super().__init__()
        self.dataset_key = 'newyorkercartoon'
        self.dataset_dir = dataset_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt = NewYorkerCartoonPrompt()

        self.image_dir = os.path.join(self.dataset_dir, 'cartoons')
        self.caption_dir = os.path.join(self.dataset_dir, 'summaries')
        self.csv_path_suffix_1 = 'LilUCB'
        self.csv_path_suffix_2 = 'lil-KLUCB'
        self.metrics = [AccuracyMetric(), PrecisionMetric(), RecallMetric(), F1ScoreMetric()]

    def load(self):
        shell_command('git clone https://github.com/nextml/caption-contest-data')

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        predictions = []
        ground_truth = []
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

            for i in range(1,5):
                captions.append(df.iloc[-1*i]['caption'])
            
            for i in range(len(captions)):
                text = self.get_prompt(captions[i])
                output = self.model.generate(text, img_path)
                answer = self.model.answer_extractor(output, self.dataset_key)

                if i == 0:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)

                if answer == 'yes':
                    predictions.append(1)
                else:
                    predictions.append(0)
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return results
    
    def evaluate_dataset_batched(self,
                                model,
                                batch_size = 32
                                ):
        self.load()
        self.model = model
        texts = []
        images = []
        ground_truth_list = []
        predictions = []
        print(len(os.listdir(self.image_dir)))
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

            for i in range(1,5):
                captions.append(df.iloc[-1*i]['caption'])
            
            for i in range(len(captions)):
                text = self.get_prompt(captions[i])
                texts.append(text)
                raw_image = Image.open(img_path).convert('RGB')
                image = self.model.get_image_tensor(raw_image)
                images.append(image)
                if i == 0:
                    ground_truth_list.append(1)
                else:
                    ground_truth_list.append(0)
        
        print(len(images))
        import pickle 
        pickle.dump(images, open("./temp.pkl", "wb"))
        print(type(images))
        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.device)
        outputs = self.model.generate_batch(images_tensor, texts, batch_size)
        for answer in outputs:
            if answer == 'yes':
                predictions.append(1)
            else:
                predictions.append(0)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth_list, predictions)
        return results    
