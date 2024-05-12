import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import json

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.slake_prompt import SlakePrompt

class SlakeDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='Slake1.0/imgs',
                annotation_file="Slake1.0/test.json",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.load()
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = SlakePrompt()
        all_annotation = json.load(open(os.path.join(download_dir, annotation_file)))
        self.annotation = []
        for ann in all_annotation:
            if ann["q_lang"] == "en" and ann["answer_type"] == "CLOSED":
                self.annotation.append(ann)

    def load(self):
        if not os.path.exists(f"{self.download_dir}/Slake1.0"):
            shell_command(f"mkdir -p {self.download_dir}/Slake1.0")
            shell_command(f"wget https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/test.json -P {self.download_dir}/Slake1.0/")
            shell_command(f"wget https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip -P {self.download_dir}/Slake1.0/")
            shell_command(f"unzip {self.download_dir}/Slake1.0/imgs.zip -d {self.download_dir}/Slake1.0/")

    def __len__(self):
        return len(self.annotation)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
  
        predictions = []
        ground_truth = []

        for row in tqdm(self.annotation, total=len(self.annotation)):
            label = row["answer"]
            question = row["question"]
            image_path = f"{self.image_dir}/{row['img_name']}"
            text = self.get_prompt(question)
            output = model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)

        return predictions, ground_truth
    
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.model = model
        
        predictions = []
        ground_truth = []

        texts = []
        images = []

        for row in tqdm(self.annotation, total=len(self.annotation)):
            label = row["answer"]
            question = row["question"]
            image_path = f"{self.image_dir}/{row['img_name']}"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt(question)
            texts.append(text)
            ground_truth.append(label)
        
        predictions = self.predict_batched(images, texts, batch_size)
        return predictions, ground_truth
