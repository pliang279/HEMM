from typing import Optional, Union

import torch
from PIL import Image
import re
from hemm.models.model import HEMMModel
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

class BLIP2(HEMMModel):
	def __init__(self,
				 device="cuda",
				 download_dir="./",
				 **kwargs,
				 ):
		super().__init__()
		self.model_type = kwargs["model_type"]
		self.device = device

	def load_weights(self):
		self.model, self.processor, _ = load_model_and_preprocess(
			name="blip2_t5", model_type=self.model_type, is_eval=True, device=self.device)

	def generate(self,
				text,
				image,):
		if not isinstance(image, Image.Image):
			image = Image.open(image).convert("RGB")
		
		processed_image = self.processor["eval"](image).unsqueeze(0).to(self.device)
		generated_text = self.model.generate({"image": processed_image, "prompt":text})[0].strip()
		return generated_text
	
	def get_image_tensor(self, image):
		img = self.processor["eval"](image).unsqueeze(0)
		return img

	def generate_batch(self, 
					   images,
					   texts, 
					   batch_size, 
					   ):
		answers = []
		for i in tqdm(range(0, len(texts), batch_size)):
			img_batch = images[i : i + batch_size].to(self.device)
			text_batch = texts[i : i + batch_size]
			generated_text = self.model.generate({"image":img_batch, "prompt":text_batch})
			answers += generated_text
				
		return answers
	