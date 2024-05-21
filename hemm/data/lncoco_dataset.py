import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator

from hemm.utils.common_utils import shell_command
from hemm.prompts.lncoco_prompt import LNCOCOPrompt

class LNCOCODatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir="ln_coco/val2017/",
                annotation_file="ln_coco/coco_val_captions.jsonl",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = LNCOCOPrompt()
        self.annotation_file = os.path.join(download_dir, annotation_file)
        self.load()

    def load(self):
        if not os.path.exists(f"{self.download_dir}/ln_coco/"):
            shell_command(f"mkdir -p {self.download_dir}/ln_coco")
            shell_command(f"wget https://storage.googleapis.com/localized-narratives/annotations/coco_val_captions.jsonl -P {self.download_dir}/ln_coco")
            shell_command(f"wget http://images.cocodataset.org/zips/val2017.zip -P {self.download_dir}/ln_coco")
            shell_command(f"unzip {self.download_dir}/ln_coco/val2017.zip -d {self.download_dir}/ln_coco")

    def __len__(self):
        with open(self.annotation_file) as f:
            annotations = f.readlines()
        return len(annotations)

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
 
        predictions = []
        ground_truth = []

        texts = []
        with open(self.annotation_file) as f:
            annotations = f.readlines()
        
        for row in tqdm(annotations, total=len(annotations)):
            ann = json.loads(row)
            img_id = ann["image_id"]
            text = self.get_prompt(ann["caption"])
            texts.append(text)
            gt_img = f"{self.image_dir}/000000{img_id}.jpg"
            pred_img = model.generate_image(text)
            predictions.append(pred_img)
            ground_truth.append(gt_img)

        return predictions, ground_truth
    
    def evaluate_dataset_batched(self, model=None, batch_size=None):
        pass
        