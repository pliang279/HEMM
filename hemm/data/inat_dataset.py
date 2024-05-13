import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import subprocess
from glob import glob
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.inat_prompt import INATPrompt
from hemm.utils.common_utils import shell_command

class INATDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='inat/val',
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = INATPrompt()
        self.load()
    
    def load(self):
        if not os.path.exists(f"{self.download_dir}/inat/"):
            shell_command(f"mkdir -p {self.download_dir}/inat")
            shell_command(f"wget  https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz -P {self.download_dir}/inat/")
            shell_command(f"tar -xvf {self.download_dir}/inat/val.tar.gz -C {self.download_dir}/inat/")

    def __len__(self):
        all_images = glob(f"{self.image_dir}/*/*.jpg")
        return len(all_images)

    def get_prompt(self) -> str:
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ):
    
        all_images = glob(f"{self.image_dir}/*/*.jpg")

        ground_truth = []
        predictions = []

        for idx in tqdm(range(len(all_images)), total=len(all_images)):
            image_path = all_images[idx]
            text = self.get_prompt()
            output = model.generate(text, image_path)

            gt_name = " ".join(all_images[idx].split("/")[-2].split("_")[-2:])
            ground_truth.append(gt_name.lower())
            predictions.append(output)

        return predictions, ground_truth
     
    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):
        self.model = model
        all_images = glob(f"{self.image_dir}/*/*.jpg")

        ground_truth = []
        predictions = []

        images = []
        texts = []
        raw_images = []
        all_preds = []

        cnt = 0
        for idx in tqdm(range(len(all_images)), total=len(all_images)):
            image_path = all_images[idx]
            text = self.get_prompt()
            raw_image = Image.open(image_path).convert("RGB")
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            texts.append(text)

            gt_name = " ".join(all_images[idx].split("/")[-2].split("_")[-2:])
            ground_truth.append(gt_name)

            if len(images) % batch_size == 0:
                predictions = self.predict_batched(images, texts, batch_size)
                all_preds += predictions
                images = []
                texts = []
                
        if len(images) > 0:
            predictions = self.predict_batched(images, texts, batch_size)
            all_preds += predictions

        return all_preds, ground_truth
        
    