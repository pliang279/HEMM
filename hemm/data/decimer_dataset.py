import os
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import pickle
import random
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.decimer_prompt import DecimerPrompt

class DecimerDatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				download_dir="./",
				dataset_dir='DECIMER/DECIMER_HDM_Dataset_Images/DECIMER_HDM_Dataset_Images',
				annotation_file='DECIMER/DECIMER_HDM_Dataset_SMILES.tsv',
				**kwargs,
				):
		super().__init__()
		self.download_dir = os.path.join(download_dir, "DECIMER")
		self.kaggle_api_path = kwargs["kaggle_api_path"]
		self.image_dir = os.path.join(download_dir, dataset_dir)
		self.prompt = DecimerPrompt()
		random.seed(0)
		self.load()
		with open(os.path.join(download_dir, annotation_file)) as f:
			self.annotations = f.readlines()
			
		self.annotations = self.annotations[1:]
		random.shuffle(self.annotations)
		self.annotations = self.annotations[-len(self.annotations) // 10:]

	def __len__(self,):
		return len(self.annotations)

	def load(self):
		os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_api_path
		if not os.path.exists(self.download_dir):
			shell_command(f'mkdir -p {self.download_dir}')
		if not os.path.exists(f'{self.download_dir}/decimer.zip'):
			shell_command(f'kaggle datasets download -d juliajakubowska/decimer -p {self.download_dir}')
			shell_command(f'unzip {self.download_dir}/decimer.zip -d {self.download_dir}')

	def get_prompt(self):
		prompt_text = self.prompt.format_prompt()
		return prompt_text

	def evaluate_dataset(self,
						 model,
						 ) -> None:
		
		predictions = []
		ground_truth = []

		for row in tqdm(self.annotations, total=len(self.annotations)):
			img_id, label = row.strip().split("\t")
			image_path = f"{self.image_dir}/{img_id}.png"
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

		for row in tqdm(self.annotations, total=len(self.annotations)):
			img_id, label = row.strip().split("\t")
			image_path = f"{self.image_dir}/{img_id}.png"
			raw_image = Image.open(image_path).convert('RGB')
			image = self.model.get_image_tensor(raw_image)
			images.append(image)
			text = self.get_prompt()
			texts.append(text)
			ground_truth.append(label)

		predictions = self.predict_batched(images, texts, batch_size)
		return predictions, ground_truth
