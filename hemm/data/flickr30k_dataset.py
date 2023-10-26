import os
import cv2
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.prompts.face_emotion_prompt import FaceEmotionPrompt
from hemm.utils.common_utils import shell_command
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class Flickr30k(Dataset):
	def __init__(self,
				 image_dir,
				 annotation_file,
				 device,
				 ):
		self.image_dir = image_dir
		self.annotation = json.load(open(annotation_file, "r"))
		self.device = device

		self.text = []
		self.image = []
		self.txt2img = {}
		self.img2txt = {}
		self.img_txt = []

		txt_id = 0
		for img_id, ann in enumerate(self.annotation):
			self.image.append(ann['image'])
			self.img2txt[img_id] = []
			for i, caption in enumerate(ann['caption']):
				self.text.append(caption)
				self.img2txt[img_id].append(txt_id)
				self.txt2img[txt_id] = img_id
				self.img_txt.append({"image": ann["image"], "caption": caption})
				txt_id += 1

	def __getitem__(self, index):
		img_text_pair = self.img_txt[index]
		img_name = f"{image_dir}/{img_text_pair["image"]}"

		caption = img_text_pair["caption"]
		prompt = f"Question: Is the following caption suitable for the given image, Answer yes or no, Caption: {caption}"

		img = Image.open(img_name)

		return {
			"image": img, 
			"prompt": prompt,
			"gt": "yes"
		}

	def __len__(self):
		return len(self.img_txt)

# class Flickr30kEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
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
