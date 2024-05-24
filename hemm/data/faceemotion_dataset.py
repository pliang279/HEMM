import os
import json
from PIL import Image
import requests
import torch
import subprocess
from tqdm import tqdm
import pandas as pd
import random

from hemm.data.dataset import HEMMDatasetEvaluator

from hemm.prompts.face_emotion_prompt import FaceEmotionPrompt
from hemm.utils.common_utils import shell_command

class FaceEmotionDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='face_emotion',
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.data_path = os.path.join(download_dir, "face_emotion")
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = FaceEmotionPrompt()
        self.choices = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.load()

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self,):
        data_dict = {}
        for fol in os.listdir(self.data_path):
            for img in os.listdir(os.path.join(self.data_path, fol)):
                data_dict[img] = fol

        return len(data_dict)
        
    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.download_dir}/fer2013.zip'):
            shell_command(f'kaggle datasets download -d msambare/fer2013 -p {self.download_dir}')
        if not os.path.exists(self.data_path):
            shell_command(f'unzip {self.download_dir}/fer2013.zip -d {self.download_dir}')
            shell_command(f'mv {self.download_dir}/test {self.download_dir}/face_emotion')

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        
        predictions = []
        ground_truth = []
        
        data_dict = {}
        for fol in os.listdir(self.data_path):
            for img in os.listdir(os.path.join(self.data_path, fol)):
                data_dict[img] = fol
        
        data_dict_list = list(data_dict.items())
        random.shuffle(data_dict_list)
        data_dict_shuffled = dict(data_dict_list)

        for img, gt in tqdm(data_dict_shuffled.items(), total=len(data_dict_shuffled.keys())):
            image_path = os.path.join(self.data_path, gt, img)
            ground_truth.append(gt)
            text = self.get_prompt()
            output = model.generate(text, image_path)
            predictions.append(output)

        return predictions, ground_truth

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.model = model
        
        ground_truth = []
        images = []
        texts = []

        data_dict = {}
        for fol in os.listdir(self.data_path):
            for img in os.listdir(os.path.join(self.data_path, fol)):
                data_dict[img] = fol
        
        data_dict_list = list(data_dict.items())
        random.shuffle(data_dict_list)
        data_dict_shuffled = dict(data_dict_list) 

        for img, gt in tqdm(data_dict_shuffled.items(), total=len(data_dict_shuffled.keys())):
            image_path = os.path.join(self.data_path, gt, img)
            ground_truth.append(gt)
            text = self.get_prompt()
            texts.append(text)
            
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)

        predictions = self.predict_batched(images, texts, batch_size)
 
        return predictions, ground_truth
    