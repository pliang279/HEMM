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

class OKVQADatasetEvaluator(HEMMDatasetEvaluator):
	def __init__(self,
				download_dir="./",
				dataset_dir='okvqa/',
				annotation_file=None,
				**kwargs,
				):
		super().__init__()
		self.download_dir = download_dir
		self.dataset_dir = os.path.join(download_dir, dataset_dir)
		self.prompt = OKVQAPrompt()
		self.load()

	def __len__(self):
		annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
		annotations = json.load(open(annotation_file, "r"))
		return len(annotations["annotations"])

	def load(self):
		if not os.path.exists(f"{self.download_dir}/okvqa"):
			shell_command(f'mkdir -p {self.download_dir}/okvqa')
		if not os.path.exists(f"{self.download_dir}/val2014.zip"):
			shell_command(f'wget http://images.cocodataset.org/zips/val2014.zip -P {self.download_dir}/okvqa/')
		if not os.path.exists(f'{self.download_dir}/okvqa/mscoco_val2014_annotations.json.zip'):
			shell_command(f'wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip -P {self.download_dir}/okvqa/')
		if not os.path.exists(f'{self.download_dir}/okvqa/OpenEnded_mscoco_val2014_questions.json.zip'):
			shell_command(f'wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip -P {self.download_dir}/okvqa/')
		if not os.path.exists(f'{self.download_dir}/okvqa/mscoco_val2014_annotations.json'):
			shell_command(f'unzip {self.download_dir}/mscoco_val2014_annotations.json.zip -d {self.download_dir}/okvqa/')
		if not os.path.exists(f'{self.download_dir}/okvqa/OpenEnded_mscoco_val2014_questions.json'):
			shell_command(f'unzip {self.download_dir}/OpenEnded_mscoco_val2014_questions.json.zip -d {self.download_dir}/okvqa')
		if not os.path.exists(f'{self.download_dir}/okvqa/val2014'):
			shell_command(f'unzip {self.download_dir}/val2014.zip -d {self.download_dir}/okvqa/')

	def get_prompt(self, question):
		return self.prompt.format_prompt(question)

	def evaluate_dataset(self,
						model,
						) -> None:
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
			image_path = os.path.join(image_dir, f"COCO_val2014_000000{ann['image_id']}.jpg")
			if not os.path.exists(image_path):
				continue
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
			output = model.generate(text, image_path)
			predictions.append(output)

			ground_truth_answer = ground_truth[i]['answers']
			multiple_gts = []
			for gt_ans in ground_truth_answer:
				multiple_gts.append(gt_ans["answer"])
			
			multiple_gts = list(set(multiple_gts))
			ground_truth_list.append(multiple_gts)


		return predictions, ground_truth_list
	
	def evaluate_dataset_batched(self,
								model,
								batch_size=32
								):
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
			image_path = os.path.join(image_dir, f"COCO_val2014_000000{ann['image_id']}.jpg")
			if not os.path.exists(image_path):
				continue
			image_ids.append(ann["image_id"])
			qs.append(qid_to_q[ann["question_id"]])
			ground_truth.append(ann)

		predictions = []
		ground_truth_list = []
		texts = []
		images = []
		raw_images = []

		for i in tqdm(range(len(image_ids)), total=len(image_ids)):
			image_path = os.path.join(image_dir, f"COCO_val2014_000000{image_ids[i]}.jpg")
			if not os.path.exists(image_path):
				continue
			raw_image = Image.open(image_path).convert('RGB')
			raw_images.append(raw_image)
			image = model.get_image_tensor(raw_image)
			images.append(image)

			text = self.get_prompt(qs[i])
			texts.append(text)

			ground_truth_answer = ground_truth[i]['answers']
			multiple_gts = []
			for gt_ans in ground_truth_answer:
				multiple_gts.append(gt_ans["answer"])
			multiple_gts = list(set(multiple_gts))
			ground_truth_list.append(multiple_gts)


		predictions = self.predict_batched(images, texts, batch_size)

		return predictions, ground_truth_list
	