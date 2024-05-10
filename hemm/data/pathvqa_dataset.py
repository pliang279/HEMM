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
from hemm.prompts.pathvqa_prompt import PathVQAPrompt
from hemm.utils.common_utils import shell_command

class PathVQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self, download_dir="./", dataset_dir=None, annotation_file=None, **kwargs):
        self.download_dir = download_dir
        self.load()
        self.prompt = PathVQAPrompt()

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def load(self):
        if not os.path.exists(f'{self.download_dir}/Backup'):
            shell_command(f'gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1G2C2_FUCyYQKCkSeCRRiTTsLDvOAjFj5 -O {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/pathvqa_images'):
            shell_command(f'unzip {self.download_dir}/Backup/pvqa.zip -d {self.download_dir}/pathvqa_images/')
    
    def __len__(self):
        annotation_path = os.path.join(self.download_dir, 'pathvqa_images','pvqa','qas','test','test_qa.pkl')
        annotation_file = pickle.load(open(annotation_path, 'rb'))

        return len(annotation_file)
        
    def evaluate_dataset(self,
                         model,
                         ) -> None:
        images_dir = os.path.join(self.download_dir, 'pathvqa_images','pvqa','images','test')
        annotation_path = os.path.join(self.download_dir, 'pathvqa_images','pvqa','qas','test','test_qa.pkl')
        annotation_file = pickle.load(open(annotation_path, 'rb'))
        
        ground_truth = []
        predictions = []
        for index, data_dict in tqdm(enumerate(annotation_file), total=len(annotation_file)):
            image_path = os.path.join(images_dir, data_dict['image'] + '.jpg')
            question = data_dict['question']
            ground_truth_answer = data_dict["answer"]
            text = self.get_prompt(question)
            output = model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
        
        return predictions, ground_truth

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):
        self.model = model
        
        images_dir = os.path.join(self.download_dir, 'pathvqa_images','pvqa','images','test')
        annotation_path = os.path.join(self.download_dir, 'pathvqa_images','pvqa','qas','test','test_qa.pkl')
        annotation_file = pickle.load(open(annotation_path, 'rb'))
        
        texts = []
        images = []

        ground_truth = []
        predictions = []
        for index, data_dict in tqdm(enumerate(annotation_file), total=len(annotation_file)):
            image_path = os.path.join(images_dir, data_dict['image'] + '.jpg')
            question = data_dict['question']
            ground_truth_answer = data_dict["answer"]
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt(question)
            texts.append(text)
            
            ground_truth.append(ground_truth_answer)
        
        samples = len(images)
        predictions = self.predict_batched(images, texts, batch_size)
        
        return predictions, ground_truth