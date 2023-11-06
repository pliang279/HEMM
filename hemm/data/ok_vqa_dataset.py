import os
import numpy as np
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.okqvqa_prompt import OKVQAPrompt
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class OKVQADatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				dataset_dir='./',
				):
		super().__init__()
		self.dataset_dir = dataset_dir
		self.dataset_key = 'okvqa'
		self.prompt = OKVQAPrompt()
		self.metrics = [BertScoreMetric(), BleuMetric()]

	def __len__(self):
		annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
		annotations = json.load(open(annotation_file, "r"))
		return len(annotations["annotations"])

	def load(self):
		if not os.path.exists('val2014.zip'):
			shell_command('wget http://images.cocodataset.org/zips/val2014.zip')
		if not os.path.exists('mscoco_val2014_annotations.json.zip'):
			shell_command('wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip')
		if not os.path.exists('OpenEnded_mscoco_val2014_questions.json.zip'):
			shell_command('wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip')
		if not os.path.exists('mscoco_val2014_annotations.json'):
			shell_command('unzip mscoco_val2014_annotations.json.zip -d ./')
		if not os.path.exists('OpenEnded_mscoco_val2014_questions.json'):
			shell_command('unzip OpenEnded_mscoco_val2014_questions.json.zip -d ./')
		if not os.path.exists('val2014'):
			shell_command('unzip val2014.zip -d ./')

	def get_prompt(self, question):
		return self.prompt.format_prompt(question)

	def evaluate_dataset(self,
						model,
						) -> None:
		self.load()
		self.model = model

		image_dir = os.path.join(self.dataset_dir, 'val2014')        
		annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
		question_file = os.path.join(self.dataset_dir, 'OpenEnded_mscoco_val2014_questions.json')

		annotations = json.load(open(annotation_file, "r"))
		questions = json.load(open(question_file, "r"))

		qid_to_q = {}
		for ques in questions["questions"]:
			qid_to_q[ques["question_id"]] = ques["question"]
		
		images = []
		qs = []
		ground_truth = []

		for ann in annotations["annotations"]:
			images.append(ann["image_id"])
			qs.append(qid_to_q[ann["question_id"]])
			ground_truth.append(ann)
		
		predictions = []
		ground_truth_list = []
		for i in tqdm(range(len(images)), total=len(images)):
			image_path = os.path.join(image_dir, f"COCO_val2014_000000{images[i]}.jpg")
			if not os.path.exists(image_path):
				continue
			text = self.get_prompt(qs[i])
			output = self.model.generate(text, image_path)
			ground_truth_answer = ground_truth[i]['answers'][0]['raw_answer']
			ground_truth_list.append(ground_truth_answer)
			predictions.append(output)

		results = {}
		for metric in self.metrics:
			results[metric.name] = metric.compute(ground_truth_list, predictions)
		return predictions, results
	
	def evaluate_dataset_batched(self,
								model,
								batch_size=32
								):
		self.load()
		self.model = model

		image_dir = os.path.join(self.dataset_dir, 'val2014')        
		annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
		question_file = os.path.join(self.dataset_dir, 'OpenEnded_mscoco_val2014_questions.json')

		annotations = json.load(open(annotation_file, "r"))
		questions = json.load(open(question_file, "r"))

		qid_to_q = {}
		for ques in questions["questions"]:
				qid_to_q[ques["question_id"]] = ques["question"]
		
		image_ids = []
		qs = []
		ground_truth = []

		for ann in annotations["annotations"]:
			image_ids.append(ann["image_id"])
			qs.append(qid_to_q[ann["question_id"]])
			ground_truth.append(ann)

		predictions = []
		ground_truth_list = []
		texts = []
		images = []
		for i in tqdm(range(len(image_ids)), total=len(image_ids)):
			image_path = os.path.join(image_dir, f"COCO_val2014_000000{image_ids[i]}.jpg")
			if not os.path.exists(image_path):
				continue
			raw_image = Image.open(image_path).convert('RGB')
			image = self.model.get_image_tensor(raw_image)
			images.append(image)

			text = self.get_prompt(qs[i])
			texts.append(text)

			ground_truth_answer = ground_truth[i]['answers'][0]['raw_answer']
			ground_truth_list.append(ground_truth_answer)

		samples = len(images) // 10
		predictions = self.predict_batched(images[:samples], texts[:samples], batch_size)

		results = {}
		for metric in self.metrics:
			results[metric.name] = metric.compute(ground_truth_list[:samples], predictions)
		
		return predictions, results, ground_truth_list[:samples]
	