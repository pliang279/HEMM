import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.plip_kather_prompt import PlipKatherPrompt
from huggingface_hub import snapshot_download

class OpenPathDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir='open_path/',
                annotation_file="open_path/Kather_test/Kather_test.csv",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.load()
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = PlipKatherPrompt()
        with open(os.path.join(download_dir, annotation_file)) as f:
            self.annotation = f.readlines()
        self.annotation = self.annotation[1:]

    def load(self):
        if not os.path.exists(f"{self.download_dir}/open_path"):
            shell_command(f"mkdir -p {self.download_dir}/open_path")
            snapshot_download(repo_id="akshayg08/OpenPath", repo_type="dataset", local_dir=f"{self.download_dir}/open_path/")

    def __len__(self):
        return len(self.annotation)

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        predictions = []
        ground_truth = []
        
        for row in tqdm(self.annotation, total=len(self.annotation)):
            _, fn, lb, caption = row.strip().split(",")
            label = " ".join(caption.split()[5:])[:-1]
            image_path = f"{self.image_dir}/{lb}/{fn}"
            text = self.get_prompt()
            
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
            _, fn, lb, caption = row.strip().split(",")
            label = " ".join(caption.split()[5:])[:-1]
            image_path = f"{self.image_dir}/{lb}/{fn}"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(label)

        predictions = self.predict_batched(images, texts, batch_size)
        return predictions, ground_truth
