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

class Kather(Dataset):
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
		_, fn, lb, caption = ann.strip().split(",")
		gt = " ".join(caption.split()[5:])[:-1]
		img_name = f"{self.image_dir}/{lb}/{fn}"

		prompt = "Choose from the below choices, Given image is a hematoxylin and eosin image of: cancer-associated stroma, adipose tissue, debris, lymphocytes, mucus, background, normal colon mucosa, colorectal adenocarcinoma epithelium, smooth muscle"
		img = np.asarray(Image.open(img_name).convert("RGB"))

		return {
			"image": img, 
			"prompt": prompt,
			"gt": gt
		}

	def __len__(self):
		return len(self.annotation)

# class KatherEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
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
