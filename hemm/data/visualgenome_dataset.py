import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
import requests

from hemm.data.dataset import HEMMDatasetEvaluator

from hemm.utils.common_utils import shell_command
from hemm.prompts.visualgenome_prompt import visualgenomeprompt

class VisualGenomeEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				download_dir="./",
				dataset_dir=None,
				annotation_file="visual_genome/question_answers.json",
				**kwargs,
				):
		super().__init__()
		self.download_dir = download_dir
		self.prompt = visualgenomeprompt()
		self.questions_json_path = os.path.join(download_dir, annotation_file)
		self.load()
	
	def load(self):
		if not os.path.exists(f"{self.download_dir}/visual_genome"):
			shell_command(f"mkdir -p {self.download_dir}/visual_genome")
			shell_command(f"wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip -P {self.download_dir}/visual_genome/")
			shell_command(f"unzip {self.download_dir}/visual_genome/question_answers.json.zip -d {self.download_dir}/visual_genome/")

	def get_prompt(self, text):
		prompt_text = self.prompt.format_prompt(text)
		return prompt_text
	
	def __len__(self):
		f = open(self.questions_json_path)
		data_vqa = json.load(f)
		return len(data_vqa)

	def evaluate_dataset(self,
						model,
						) -> None:
		predictions = []
		ground_truth = []
		f = open(self.questions_json_path)
		data_vqa = json.load(f)
		for i in tqdm(range(len(data_vqa)), total=len(data_vqa)):
			temp_dict=data_vqa[i]
			img_id=temp_dict['id']
			qas=temp_dict['qas']
			try:
				if i==1:
					url=f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_id}.jpg"
					image=Image.open(requests.get(url, stream=True).raw)
					image_b = image.resize((640,480))
				else:
					url=f"https://cs.stanford.edu/people/rak248/VG_100K/{img_id}.jpg"
					image=Image.open(requests.get(url, stream=True).raw)
					image_b = image.resize((640,480))
			except:
				continue

			for j in range(len(qas)):
				question=qas[j]['question']
				question_pmt=self.get_prompt(question)
				
				output = model.generate(question_pmt, image_b)
				predictions.append(output)
				ground_truth.append(qas[j]['answer'])

		return predictions, ground_truth

	def evaluate_dataset_batched(self, model, batch_size=32):
		self.model = model

		images = []
		texts = []
		predictions = []
		ground_truth = []
		all_preds = []

		f = open(self.questions_json_path)

		data_vqa = json.load(f)
		for i in tqdm(range(len(data_vqa)), total = len(data_vqa)):
			temp_dict=data_vqa[i]
			img_id=temp_dict['id']
			qas=temp_dict['qas']
			try:
				if i==1:
					url=f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_id}.jpg"
					image=Image.open(requests.get(url, stream=True).raw)
					image_b = image.resize((640,480))
				else:
					url=f"https://cs.stanford.edu/people/rak248/VG_100K/{img_id}.jpg"
					image=Image.open(requests.get(url, stream=True).raw)
					image_b = image.resize((640,480))
			except:
				continue
			
			for j in range(len(qas)):
				question=qas[j]['question']
				question_pmt=self.get_prompt(question)
				texts.append(question_pmt)
				images.append(self.model.get_image_tensor(image_b))
				ground_truth.append(qas[j]['answer'])

			if len(images) % batch_size == 0:
				predictions = self.predict_batched(images, texts, batch_size)
				all_preds += predictions
				images = []
				texts = []
			
		if len(images) > 0:
			predictions = self.predict_batched(images, texts, batch_size)
			all_preds += predictions

		return all_preds, ground_truth
	