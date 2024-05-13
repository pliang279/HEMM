import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import random
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.enrico_prompt import EnricoPrompt

class EnricoDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='enrico/screenshots',
                annotation_file="enrico/design_topics.csv",
                **kwargs, 
                 ):
        super().__init__()
        self.download_dir = os.path.join(download_dir, "enrico")
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = EnricoPrompt()
        random.seed(0) 
        self.load()
        with open(os.path.join(download_dir, annotation_file)) as f:
            self.annotations = f.readlines()
        
        self.annotations = self.annotations[1:]
        random.shuffle(self.annotations)
        self.annotations = self.annotations[-len(self.annotations) // 10:]
     
    def __len__(self,):
        return len(self.annotations)

    def load(self):
        if not os.path.exists(self.download_dir):
            shell_command(f'mkdir -p {self.download_dir}')
            shell_command(f'wget http://userinterfaces.aalto.fi/enrico/resources/screenshots.zip -P {self.download_dir}')
            shell_command(f'unzip {self.download_dir}/screenshots.zip -d {self.download_dir}')
            shell_command(f"wget https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv -P {self.download_dir}")

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load() 
        predictions = []
        ground_truth = []

        for row in tqdm(self.annotations, total=len(self.annotations)):
            img_id, label = row.strip().split(",")
            image_path = f"{self.image_dir}/{img_id}.jpg"
            text = self.get_prompt()
            output = model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)
        return predictions, ground_truth
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []

        texts = []
        images = []

        for row in tqdm(self.annotations, total=len(self.annotations)):
            img_id, label = row.strip().split(",")
            image_path = f"{self.image_dir}/{img_id}.jpg"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(label)
        
        predictions = self.predict_batched(images, texts, batch_size)
        return predictions, ground_truth
