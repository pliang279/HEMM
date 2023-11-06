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
from hemm.metrics.metric import HEMMMetric
from hemm.prompts.face_emotion_prompt import FaceEmotionPrompt
from hemm.utils.common_utils import shell_command
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class FaceEmotionDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 data_path = 'face_emotion',
                 kaggle_api_path = None
                 ):
        super().__init__()
        self.dataset_key = 'face_emotion'
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kaggle_api_path = kaggle_api_path
        self.prompt = FaceEmotionPrompt()
        self.choices = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.metrics = [BertScoreMetric(), BleuMetric()]

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text
    
    def __len__(self,):
        data_dict = {}
        for fol in os.listdir(self.data_path):
            for img in os.listdir(os.path.join(self.data_path, fol)):
                data_dict[img] = fol

        return len(data_dict)
        

    def load(self, kaggle_api_path):
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_api_path
        if not os.path.exists('fer2013.zip'):
          shell_command('kaggle datasets download -d msambare/fer2013')
        if not os.path.exists('face_emotion'):
          shell_command('unzip fer2013.zip')
          shell_command('mv test face_emotion')

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.model = model
        
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
            output = self.model.generate(text, image_path)
            predictions.append(output)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.load(self.kaggle_api_path)
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

        samples = len(images) // 10
        predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth[:samples], predictions)
    
        return predictions, results, ground_truth[:samples]
    