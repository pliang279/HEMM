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

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def load(self, kaggle_api_path):
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_api_path
        if not os.path.exists('fer2013.zip'):
          shell_command('kaggle datasets download -d msambare/fer2013')
        if not os.path.exists('face_emotion'):
          shell_command('unzip fer2013.zip')
          shell_command('mv test face_emotion')

    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.metric = metric
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
            ground_truth.append(self.choices.index(gt))
            text = self.get_prompt()
            output = self.model.generate(text, image_path)
            predictions.append(output)

        results = self.metric.compute(ground_truth, predictions)
        return results

    def evaluate_dataset_batched(self,
                         model,
                         metric,
                         batch_size=32
                         ) -> None:
        self.load(self.kaggle_api_path)
        self.metric = metric
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
            ground_truth.append(self.choices.index(gt))
            text = self.get_prompt()
            texts.append(text)
            
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)

        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.chat.device)
        outputs = self.model.generate_batch(images_tensor, texts, batch_size)

        results = self.metric.compute(ground_truth, outputs)
        return results