import os
import json
from PIL import Image, ImageFile
import requests
import torch
import subprocess
from tqdm import tqdm
import pandas as pd
import random 

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.memotion_prompt import MemotionPrompt
from hemm.utils.common_utils import shell_command

ImageFile.LOAD_TRUNCATED_IMAGES = True

def ref_text(text):
    sents = text.split("\n")
    sents = [sent.strip() for sent in sents]
    return " ".join(sents).strip()

class MemotionDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                download_dir="./",
                dataset_dir=None,
                annotation_file=None,
                **kwargs,
                ):
        super().__init__()
        self.download_dir = download_dir
        self.data_path = os.path.join(download_dir, 'memotion-dataset-7k/memotion_dataset_7k/labels.xlsx')
        self.image_dir = os.path.join(download_dir, 'memotion-dataset-7k/memotion_dataset_7k/images')
        self.kaggle_api_path = kwargs["kaggle_api_path"]
        self.prompt = MemotionPrompt()
        self.choices = ['funny', 'very_funny', 'not_funny', 'hilarious']
        self.load()

    def __len__(self):
        df = pd.read_excel(self.data_path)
        return len(df)

    def get_prompt(self, caption) -> str:
        prompt_text = self.prompt.format_prompt(caption)
        return prompt_text

    def load(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
        if not os.path.exists(f'{self.download_dir}/memotion-dataset-7k.zip'):
          shell_command(f'kaggle datasets download -d williamscott701/memotion-dataset-7k -p {self.download_dir}')
        if not os.path.exists(f'{self.download_dir}/memotion-dataset-7k'):
          shell_command(f'unzip {self.download_dir}/memotion-dataset-7k.zip -d {self.download_dir}/memotion-dataset-7k')

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        df = pd.read_excel(self.data_path)
        predictions = []
        ground_truth = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(self.image_dir, row['image_name'])
            caption = row['text_corrected']
            gt_label = row['humour']
            ground_truth.append(gt_label)
            text = self.get_prompt(caption)
            output = model.generate(text, image_path)
            predictions.append(output)

        return predictions, ground_truth

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ):
        self.model = model
        df = pd.read_excel(self.data_path)
        ground_truth = []
        images = []
        texts = []

        for index, row in tqdm(df.iterrows(), total=len(df)):
            image_path = os.path.join(self.image_dir, row['image_name'])
            caption = row['text_corrected']
            gt_label = row['humour']
            ground_truth.append(gt_label)

            raw_image = Image.open(image_path).convert('RGB')

            image = self.model.get_image_tensor(raw_image)
            images.append(image)

            text = self.get_prompt(caption)
            texts.append(text)

        predictions = self.predict_batched(images, texts, batch_size)
        return predictions, ground_truth
    