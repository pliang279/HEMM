import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.decimer_prompt import DecimerPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class DecimerDatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 image_dir='/work/agoindan/DECIMER/DECIMER_HDM_Dataset_Images',
                 annotation_file='/work/agoindan/DECIMER/DECIMER_HDM_Dataset_SMILES.tsv',
                 device="cpu",
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.prompt = DecimerPrompt()
        self.annotation_file = annotation_file
        self.metrics = [BertScoreMetric(), BleuMetric()]

    # def load(self):
    #   os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
    #   if not os.path.exists('landuse-scene-classification.zip'):
    #       shell_command('kaggle datasets download -d apollo2506/landuse-scene-classification')
    #   if not os.path.exists('ucmercedimages'):
    #       shell_command('unzip landuse-scene-classification.zip -d ucmercedimages/')

    def load(self):
        pass

    def get_prompt(self):
        prompt_text = self.prompt.format_prompt()
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        self.load()
        self.model = model
        
        predictions = []
        ground_truth = []

        with open(self.annotation_file) as f:
            annotations = f.readlines()
        annotations = annotations[1:]

        for row in tqdm(annotations, total=len(annotations)):
            img_id, label = row.strip().split("\t")
            image_path = f"{self.image_dir}/{img_id}.png"
            text = self.get_prompt()
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(label)
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results
    
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

        with open(self.annotation_file) as f:
            annotations = f.readlines()
        annotations = annotations[1:]

        for row in tqdm(annotations, total=len(annotations)):
            img_id, label = row.strip().split("\t")
            image_path = f"{self.image_dir}/{img_id}.png"
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)
            text = self.get_prompt()
            texts.append(text)
            ground_truth.append(label)

        predictions = self.predict_batched(images, texts, batch_size)
        
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(ground_truth, predictions)
        return predictions, results
        