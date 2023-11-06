import os
import cv2
import random
import json
from typing import Optional, Union, List
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.prompts.vcr_prompt import VCRPrompt
from hemm.utils.common_utils import shell_command
from ast import literal_eval
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class VCRDatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				):
		super().__init__()
		self.annotation_file = 'vcr_annotations/val.jsonl'
		self.image_dir = 'vcr_images/vcr1images/'
		self.prompt = VCRPrompt()
		self.dataset_key = 'vcr'
		self.metrics = [BertScoreMetric(), BleuMetric()]
		self.load()

	def __len__(self):
		with open(self.annotation_file) as f:
			self.annotations = f.readlines()
		
		return len(self.annotations)

	def load(self):
		if not os.path.exists('vcr1images.zip'):
			shell_command('wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip')
		if not os.path.exists('vcr1annots.zip'):
			shell_command('wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1annots.zip')
		if not os.path.exists('vcr_annotations'):
			shell_command('unzip vcr1annots.zip -d vcr_annotations')
		if not os.path.exists('vcr_images'):
			shell_command('unzip vcr1images.zip -d vcr_images')

	def get_prompt(self, question, answer_choices, rationale_choices=None, answer_label=None):
		prompt = self.prompt.format_prompt(question, answer_choices, rationale_choices, answer_label)
		return prompt
	
	def draw_segments(self, image, segments, color=(0, 0, 255), thickness=2):
		for segment in segments:
			if len(segment) == 0:
				continue
			for i in range(len(segment[0]) - 1):
				x1, y1 = segment[0][i]
				x2, y2 = segment[0][i + 1]
				cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
	
	def draw_bounding_boxes(self, image, boxes, names, color=(0, 255, 0), thickness=2, font_scale=0.6):
		for box, name in zip(boxes, names):
			x1, y1, x2, y2, _ = box
			cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
			cv2.putText(image, name, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

	def fix_tokenization(self, tokenized_sent, obj_to_type):
		"""
		Turn a detection list into what we want: some text.
		:param tokenized_sent: Tokenized sentence with detections collapsed to a list.
		:param obj_to_type: [person, person, pottedplant] 
		:return: tokenized sentence
		"""

		new_tokenization = []
		for tok in tokenized_sent:
			if isinstance(tok, list):
				names = []
				for int_name in tok:
					obj_type = obj_to_type[int_name]
					ind = int_name
					text_to_use = f"{obj_type}{ind}"
					names.append(text_to_use)

				new_tokenization.append(", ".join(names))
			else:
				new_tokenization.append(tok)
	
		return " ".join(new_tokenization)

	def evaluate_dataset(self,
						 model,
						 ) -> None:
		# self.load()
		self.model = model

		with open(self.annotation_file) as f:
			self.annotations = f.readlines()

		predictions = []
		ground_truth = []
		outputs = []
		for i in tqdm(range(len(self.annotations)), total=len(self.annotations)):
			ann = literal_eval(self.annotations[i])
			question = self.fix_tokenization(ann["question"], ann["objects"])
			new_answer_choices = []
			for ch in ann["answer_choices"]:
				new_answer_choices.append(self.fix_tokenization(ch, ann["objects"]))
			new_rationale_choices = []
			for ch in ann["rationale_choices"]:
				new_rationale_choices.append(self.fix_tokenization(ch, ann["objects"]))
			
			prompt = self.get_prompt(question, new_answer_choices, new_rationale_choices, ann["answer_label"])
			img = np.asarray(Image.open(os.path.join(self.image_dir, ann['img_fn'])).convert("RGB"))
			metadata = json.load(open(os.path.join(self.image_dir, ann['metadata_fn'])))

			boxes = metadata["boxes"]
			segments = metadata["segms"]

			names = []
			for idx, obj in enumerate(ann["objects"]):
				names.append(f"{obj}{idx}")
			
			self.draw_bounding_boxes(img, boxes, names)
			self.draw_segments(img, segments)

			image_path = './current_image.jpg'
			img = Image.fromarray(img)
			img = img.save(image_path)
			output = self.model.generate(prompt, image_path)
			outputs.append(output)
			answer = self.model.answer_extractor(output, self.dataset_key)
			ground_truth.append(ann["answer_label"])
			predictions.append(answer)
			
		results = {}
		for metric in self.metrics:
			results[metric.name] = metric.compute(ground_truth, predictions)
			
		return outputs, results

	def evaluate_dataset_batched(self,
						 model,
						 batch_size = 32
						 ):
		self.load()
		self.model = model

		texts = []
		images = []

		with open(self.annotation_file) as f:
			self.annotations = f.readlines()

		ground_truth = []
		for i in tqdm(range(len(self.annotations)), total=len(self.annotations)):
			ann = literal_eval(self.annotations[i])
			question = self.fix_tokenization(ann["question"], ann["objects"])
			new_answer_choices = []
			for ch in ann["answer_choices"]:
				new_answer_choices.append(self.fix_tokenization(ch, ann["objects"]))
			new_rationale_choices = []
			for ch in ann["rationale_choices"]:
				new_rationale_choices.append(self.fix_tokenization(ch, ann["objects"]))
			
			prompt = self.get_prompt(question, new_answer_choices, new_rationale_choices, ann["answer_label"])
			img = np.asarray(Image.open(os.path.join(self.image_dir, ann['img_fn'])).convert("RGB"))
			metadata = json.load(open(os.path.join(self.image_dir, ann['metadata_fn'])))

			boxes = metadata["boxes"]
			segments = metadata["segms"]

			names = []
			for idx, obj in enumerate(ann["objects"]):
				names.append(f"{obj}{idx}")
			
			self.draw_bounding_boxes(img, boxes, names)
			self.draw_segments(img, segments)

			image_path = './current_image.jpg'
			img = Image.fromarray(img)
			img = img.save(image_path)
			raw_image = Image.open(image_path).convert('RGB')
			image = self.model.get_image_tensor(raw_image)
			images.append(image)
			texts.append(prompt)
			
			ground_truth.append(ann["answer_label"])
			
		# images_tensor = torch.cat(images, dim=0)
		# images_tensor = images_tensor.to(self.model.device)
		# predictions = self.model.generate_batch(images_tensor, texts, batch_size)

		samples = len(images) // 10
		predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)

		results = {}
		for metric in self.metrics:
			results[metric.name] = metric.compute(ground_truth[:samples], predictions)
		return predictions, results, ground_truth[:samples]
	