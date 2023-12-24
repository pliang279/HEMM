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
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class MemeCapsDatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				 annotation_path = 'memes-test.json',
				 images = 'memecap_images/memes',
				 ):
		self.annotation_path = annotation_path
		self.images = images
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.prompt = MemeCapsPrompt()
		self.metrics = [BertScoreMetric(), BleuMetric()]

	def get_prompt(self, title, image_description) -> str:
		prompt_text = self.prompt.format_prompt(title, image_description)
		return prompt_text
	
	def __len__(self):
		annotation_file = json.load(open(self.annotation_path))
		return len(annotation_file)

	def load(self):
		if not os.path.exists('memes.zip'):
			shell_command('gdown https://drive.google.com/uc?id=1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1')
		if not os.path.exists('memes-test.json'):
			shell_command('wget https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-test.json')
		if not os.path.exists('memecap_images/'):
			shell_command('unzip memes.zip -d memecap_images/')
		
	def evaluate_dataset(self,
						 model,
						 ) -> None:
		self.load()
		self.model = model
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
			output = self.model.generate(text, image_path)
			predictions.append(output)

		return predictions, ground_truth

	def evaluate_dataset_batched(self,
						 model,
						 batch_size=32
						 ) -> None:
		self.load()
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

		samples = len(images)
		predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)
		# print(len(raw_images))
		# samples = len(raw_images)
		# self.save_details(raw_images[:samples], texts[:samples], ground_truth[:samples], "memecaps.pkl")	
		
		return predictions, ground_truth[:samples]
	