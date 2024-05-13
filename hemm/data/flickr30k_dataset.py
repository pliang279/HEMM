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
from hemm.prompts.flickr30k_prompt import Flickr30kPrompt
from hemm.utils.common_utils import shell_command

class Flickr30kDatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				download_dir="./",
				dataset_dir="flickr30k_images/flickr30k_images/",
				annotation_file="flickr30k_images/flickr30k_test.json",
				**kwargs,
				 ):
		super().__init__()
		self.download_dir = download_dir
		self.kaggle_api_path = kwargs["kaggle_api_path"]
		self.load()
		self.image_dir = os.path.join(self.download_dir, dataset_dir)
		annotation_file = os.path.join(self.download_dir, annotation_file)
		self.annotation = json.load(open(annotation_file, "r"))
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.prompt = Flickr30kPrompt()

	def get_prompt(self) -> str:
		prompt_text = self.prompt.format_prompt()
		return prompt_text
	
	def __len__(self,):
		return len(self.annotation)

	def load(self):
		os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
		if not os.path.exists(f'{self.download_dir}/flickr-image-dataset.zip'):
			shell_command(f'kaggle datasets download -d hsankesara/flickr-image-dataset -p {self.download_dir}')
		if not os.path.exists(f'{self.download_dir}/flickr30k_images'):
			shell_command(f'unzip {self.download_dir}/flickr-image-dataset.zip -d {self.download_dir}')
			shell_command(f"wget https://huggingface.co/datasets/akshayg08/Flickr30k_test/raw/main/flickr30k_test.json -P {self.download_dir}/flickr30k_images/")

	def evaluate_dataset(self,
						 model,
						 ):
		predictions = []
		ground_truth = []

		for i, ann in tqdm(enumerate(self.annotation), total=len(self.annotation)):
			image_path = f"{self.image_dir}/{ann['image'].split('/')[-1]}"
			ground_truth.append(ann["caption"][0])
			text = self.get_prompt()
			output = model.generate(text, image_path)
			predictions.append(output)
			
		return predictions, ground_truth

	def evaluate_dataset_batched(self,
						 model,
						 batch_size=32
						 ):
		self.model = model
		
		ground_truth = []
		images = []
		texts = []

		for i, ann in tqdm(enumerate(self.annotation), total=len(self.annotation)):
			image_path = f"{self.image_dir}/{ann['image'].split('/')[-1]}"
			
			raw_image = Image.open(image_path).convert('RGB')
			image = self.model.get_image_tensor(raw_image)
			images.append(image)
			ground_truth.append(ann["caption"])

			text = self.get_prompt()
			texts.append(text)

		predictions = self.predict_batched(images, texts, batch_size)
		return predictions, ground_truth
	