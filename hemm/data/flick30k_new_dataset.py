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
from hemm.prompts.flickr30k_prompt import Flickr30kPrompt
from hemm.utils.common_utils import shell_command
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class Flickr30kDatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				 image_dir="/work/agoindan/flickr30k_images/flickr30k_images/",
				 annotation_file="/work/agoindan/flickr30k_images/flickr30k_test.json",
				 kaggle_api_path="/home/agoindan/kaggle.json"
				 ):
		super().__init__()
		self.image_dir = image_dir
		self.annotation = json.load(open(annotation_file, "r"))
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.prompt = Flickr30kPrompt()
		self.metrics = [BertScoreMetric(), BleuMetric()]
		self.kaggle_api_path = kaggle_api_path

	def get_prompt(self) -> str:
		prompt_text = self.prompt.format_prompt()
		return prompt_text

	def load(self):
		os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
		if not os.path.exists('flickr-image-dataset.zip'):
			shell_command('kaggle datasets download -d hsankesara/flickr-image-dataset')
		if not os.path.exists('flickr30k_images'):
			shell_command('unzip flickr-image-dataset.zip -d flickr30k_images/')

	def evaluate_dataset(self,
						 model,
						 ) -> None:
		self.model = model
		
		predictions = []
		ground_truth = []

		for i, ann in tqdm(enumerate(self.annotation), total=len(self.annotation)):
			if i == 8:
				break
			image_path = f"{self.image_dir}/{ann['image'].split('/')[-1]}"
			ground_truth.append(ann["caption"][0])
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
		# self.load()
		self.model = model
		
		ground_truth = []
		images = []
		texts = []

		for i, ann in tqdm(enumerate(self.annotation), total=len(self.annotation)):
			image_path = f"{self.image_dir}/{ann['image'].split('/')[-1]}"
			
			raw_image = Image.open(image_path).convert('RGB')
			image = self.model.get_image_tensor(raw_image)
			images.append(image)
			ground_truth.append(ann["caption"][0])

			text = self.get_prompt()
			texts.append(text)

		predictions = self.predict_batched(images, texts, batch_size)

		results = {}
		for metric in self.metrics:
			results[metric.name] = metric.compute(ground_truth, predictions)

		return predictions, results
	