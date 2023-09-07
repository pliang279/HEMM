import os
import cv2
import json
import numpy as np
from typing import Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset

class SlakeVQA(Dataset):
	def __init__(self,
				 image_dir,
				 vis_processor, 
				 annotation_file,
				 device,
				 ):
		self.image_dir = image_dir
		all_annotation = json.load(open(annotation_file))
		self.annotation = []
		for ann in all_annotation:
			if ann["q_lang"] == "en" and ann["answer_type"] == "CLOSED":
				self.annotation.append(ann)

		self.vis_processor = vis_processor
		self.device = device

	def __getitem__(self, index):
		ann = self.annotation[index]

		gt = ann["answer"]
		question = ann["question"]
		img_name = f"{self.image_dir}/{ann['img_name']}"

		prompt = f"Answer the question in a single word, Question: {question}"
		img = Image.open(img_name).convert("RGB")
		img = self.vis_processor["eval"](img).to(self.device)

		return {
			"image": img, 
			"prompt": prompt,
			"gt": gt
		}

	def __len__(self):
		return len(self.annotation)
