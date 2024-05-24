import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator

from hemm.utils.common_utils import shell_command
from hemm.prompts.magic_brush_prompt import MagicBrushPrompt
from huggingface_hub import snapshot_download

class MagicBrushDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir="magic_brush/images",
                annotation_file="magic_brush/edit_sessions.json",
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.image_dir = os.path.join(download_dir, dataset_dir)
        self.prompt = MagicBrushPrompt()
        self.annotation_file = os.path.join(download_dir, annotation_file)
        self.load()

    def load(self):
        if not os.path.exists(f"{self.download_dir}/magic_brush/"):
            shell_command(f"mkdir -p {self.download_dir}/magic_brush")
            snapshot_download(repo_id="akshayg08/MagicBrushTest", repo_type="dataset", local_dir=f"{self.download_dir}/magic_brush/")

    def __len__(self):
        annotations = json.load(open(self.annotation_file))
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
        annotations = json.load(open(self.annotation_file))
        
        cnt = 0
        for img_id in tqdm(annotations, total=len(annotations)):
            ann = annotations[img_id]
            for sample in ann:
                input_img = f"{self.image_dir}/{img_id}/{sample['input']}"
                text = self.get_prompt(sample['instruction'])
                texts.append(text)
                gt_img = f"{self.image_dir}/{img_id}/{sample['output']}"
                pred_img = model.generate_image(text, input_img)
                predictions.append(pred_img)
                ground_truth.append(gt_img)
                cnt += 1
                if cnt == 100:
                    break
            
            if cnt == 100:
                break

        return predictions, ground_truth
    
    def evaluate_dataset_batched(self, model=None, batch_size=None):
        pass
        