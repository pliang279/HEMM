import os
import cv2
import json
import numpy as np
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
# from hemm.data.dataset import HEMMDatasetEvaluator
# from hemm.metrics.metric import HEMMMetric
# from hemm.utils.evaluator_mixin import EvaluatorMixin

class Memotion(Dataset):
	def __init__(self,
				 image_dir,
				 annotation_file,
				 device,
				 ):
		self.image_dir = image_dir
		with open(annotation_file) as f:
			self.annotation = f.readlines()
		
		self.annotation = self.annotation[1:]
		self.device = device

	def __getitem__(self, index):
		ann = self.annotation[index]
		fields = ann.strip().split(",")

		img_name = f"{self.image_dir}/{fields[1].strip()}"
		caption = fields[3].strip().lower()
		humour_label = fields[4].strip()

		prompt = f"Question: Given the Meme and the following caption, is the meme 0) funny 1) very funny 2) not funny 3) hilarious, Caption:{caption}"

		img = np.asarray(Image.open(img_name).convert("RGB"))

		return {
			"image": img, 
			"prompt": prompt,
			"gt": humour_label
		}

	def __len__(self):
		return len(self.annotation)

# class MemotionEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
# 	def __init__(self,
# 				 dataset_dir,
# 				 model,
# 				 evaluate_path,
# 				 device,
# 				 batch_size,
# 				 shuffle_dataset,
# 				 output_file_path
# 				 ):
# 		super().__init__(dataset_dir)
# 		self.dataset_dir = dataset_dir
# 		self.model = model
# 		self.evaluate_path = evaluate_path
# 		self.device = device
# 		self.batch_size = batch_size
# 		self.shuffle_dataset = shuffle_dataset
# 		self.output_file_path = output_file_path

# 	def evaluate_dataset(self,
# 						 metrics: List[HEMMMetric],
# 						 ) -> None:

# 		image_dir = os.path.join(self.dataset_dir, 'val2014')        
# 		annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
# 		question_file = os.path.join(self.dataset_dir, 'OpenEnded_mscoco_val2014_questions.json')

# 		pt_dataset = OKVQA(image_dir, annotation_file, question_file, self.device)
# 		loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
# 		self.evaluate(self.model, loader, self.output_file_path, modalities=['img','text'])
