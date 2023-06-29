import os
import cv2
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.utils.evaluator_mixin import EvaluatorMixin

def fix_tokenization(tokenized_sent, obj_to_type):
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

def generate_prompt(question, answer_choices, rationale_choices, answer_label=None):
	answer_prompt = f"Question: {question} Choose from the below choices: 0) {answer_choices[0]} 1) {answer_choices[1]} 2) {answer_choices[2]} 3) {answer_choices[3]}"
	rationale_prompt = None
	if answer_label is not None:
		rationale_prompt = answer_prompt + f"Answer: {answer_label}. Question: {answer_label} is correct because? Choose from the below choices: 0) {rationale_choices[0]} 1) {rationale_choices[1]} 2) {rationale_choices[2]} 3) {rationale_choices[3]}"

	return answer_prompt, rationale_prompt

def draw_bounding_boxes(image, boxes, names, color=(0, 255, 0), thickness=2, font_scale=0.6):
    for box, name in zip(boxes, names):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(image, name, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_segments(image, segments, color=(0, 0, 255), thickness=2):
    for segment in segments:
        for i in range(len(segment) - 1):
            x1, y1 = segment[i]
            x2, y2 = segment[i + 1]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

class VCR(Dataset):
	def __init__(self,
				 image_dir,
				 annotation_file,
				 device,
				 ):
		self.image_dir = image_dir
		with open(annotation_file) as f:
			self.annotations = f.readlines()
		self.device = device

	def __getitem__(self, index):
		ann = self.annotations[index]
		img_name = f"{image_dir}/{ann["img_fn"]}"
		question = fix_tokenization(ann["question"], , ann["objects"])
		new_answer_choices = []
		for ch in ann["answer_choices"]:
			new_answer_choices.append(fix_tokenization(ch))
		new_rationale_choices = []
		for ch in ann["rationale_choices"]:
			new_rationale_choices.append(fix_tokenization(ch))
		
		prompt, rationale_prompt = generate_prompt(question, new_answer_choices, new_rationale_choices, ann["answer_label"])

		img = Image.open(f"{self.image_dir}/{ann['img_fn']}").convert("RGB")
		metadata = json.load(open(f"{self.image_dir}/{ann['metadata_fn']}"))
		boxes = metadata["boxes"]
		segments = metadata["segments"]

		names = []
		for idx, obj in enumerate(ann["objects"]):
			names.append(f"{obj}{idx}")
		
		draw_bounding_boxes(img, boxes)
		draw_segments(img, segments)

		return {
			"image": img, 
			"prompt": prompt,
			"rationale_prompt": rationale_prompt,
			"gt": ann["answer_choice"]
		}

	def __len__(self):
		return len(self.annotations)

class VCREvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
	def __init__(self,
				 dataset_dir,
				 model,
				 evaluate_path,
				 device,
				 batch_size,
				 shuffle_dataset,
				 output_file_path
				 ):
		super().__init__(dataset_dir)
		self.dataset_dir = dataset_dir
		self.model = model
		self.evaluate_path = evaluate_path
		self.device = device
		self.batch_size = batch_size
		self.shuffle_dataset = shuffle_dataset
		self.output_file_path = output_file_path

	def evaluate_dataset(self,
						 metrics: List[HEMMMetric],
						 ) -> None:

		image_dir = os.path.join(self.dataset_dir, 'val2014')        
		annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
		question_file = os.path.join(self.dataset_dir, 'OpenEnded_mscoco_val2014_questions.json')

		pt_dataset = OKVQA(image_dir, annotation_file, question_file, self.device)
		loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
		self.evaluate(self.model, loader, self.output_file_path, modalities=['img','text'])
