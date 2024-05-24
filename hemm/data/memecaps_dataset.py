import os
import json
from typing import Optional, Union, List
from PIL import Image
import requests
import torch
from datasets import load_dataset
import subprocess
from tqdm import tqdm
from hemm.data.dataset import HEMMDatasetEvaluator

from hemm.metrics.metric import HEMMMetric
from hemm.prompts.memecaps_prompt import MemeCapsPrompt
from hemm.utils.common_utils import shell_command

class MemeCapsDatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				download_dir="./",
				images='memecap_images/memes',
				annotation_path='memes-test.json',
				**kwargs):

		self.download_dir = download_dir
		self.annotation_path = os.path.join(download_dir, annotation_path)
		self.images = os.path.join(download_dir, images)
		self.prompt = MemeCapsPrompt()
		self.load()

	def get_prompt(self, title, image_description) -> str:
		prompt_text = self.prompt.format_prompt(title, image_description)
		return prompt_text
	
	def __len__(self):
		annotation_file = json.load(open(self.annotation_path))
		return len(annotation_file)

	def load(self):
		if not os.path.exists(f'{self.download_dir}/memes.zip'):
			shell_command(f'gdown https://drive.google.com/uc?id=1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1 -O {self.download_dir}')
		if not os.path.exists(f'{self.download_dir}/memes-test.json'):
			shell_command(f'wget https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-test.json -P {self.download_dir}')
		if not os.path.exists(f'{self.download_dir}/memecap_images/'):
			shell_command(f'unzip {self.download_dir}/memes.zip -d {self.download_dir}/memecap_images/')
		
	def evaluate_dataset(self,
						 model,
						 ) -> None:

		annotation_file = json.load(open(self.annotation_path))
		predictions = []
		ground_truth = []

		for index, data_dict in tqdm(enumerate(annotation_file), total=len(annotation_file)):
			image_path = f"{self.images}/{data_dict['img_fname'].strip()}"
			image_desc = data_dict["img_captions"][0]
			title = data_dict["title"]
			gt_caption = data_dict["meme_captions"][0]
			text = self.get_prompt(title, image_desc)
			ground_truth.append(gt_caption)
			output = model.generate(text, image_path)
			predictions.append(output)
				
		return predictions, ground_truth

	def evaluate_dataset_batched(self,
						 model,
						 batch_size=32
						 ) -> None:

		self.model = model
		annotation_file = json.load(open(self.annotation_path))
		predictions = []
		ground_truth = []
		images = []
		texts = []
		raw_images = []
		for index, data_dict in tqdm(enumerate(annotation_file), total=len(annotation_file)):
			image_path = f"{self.images}/{data_dict['img_fname'].strip()}"
			raw_image = Image.open(image_path).convert('RGB')
			
			image = self.model.get_image_tensor(raw_image)
			images.append(image)
			
			image_desc = data_dict["img_captions"][0]
			title = data_dict["title"]
			gt_caption = data_dict["meme_captions"][0]
			text = self.get_prompt(title, image_desc)
			ground_truth.append(gt_caption)
			texts.append(text)

		predictions = self.predict_batched(images, texts, batch_size)		
		return predictions, ground_truth
	