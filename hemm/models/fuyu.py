import re
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from transformers import FuyuProcessor, FuyuForCausalLM

class Fuyu(HEMMModel):
	def __init__(self, device="cuda", download_dir="./", **kwargs):
		super().__init__()
		self.device = torch.device(device)

	def load_weights(self):	
		model_id = "adept/fuyu-8b"
		self.processor = FuyuProcessor.from_pretrained(model_id)
		self.model = FuyuForCausalLM.from_pretrained(model_id, device_map=self.device)

	def generate(self,
				text: Optional[str],
				image,
			) -> str:

		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image
		
		inputs = self.processor(text=text, images=raw_image, return_tensors="pt").to(self.device)
		generation_output = self.model.generate(**inputs, max_new_tokens=100)
		generation_text = self.processor.batch_decode(generation_output[:, -100:], skip_special_tokens=True)
		prediction = generation_text[0].split("\x04")[-1].strip()
		
		return prediction
			
	def get_image_tensor(self, image):
		return image	
	
	def answer_extractor(self, text, dataset_key):
		return text

	def generate_batch(self, 
					   images,
					   texts, 
					   batch_size, 
					   ):
		answers = []
		for i in tqdm(range(0, len(texts), batch_size)):
			img_batch = images[i : i + batch_size]
			text_batch = texts[i : i + batch_size]

			inputs = self.processor(text=text_batch, images=img_batch, return_tensors="pt", 
						   			padding=True, truncation=True).to(self.device)
			
			generated_output = self.model.generate(**inputs, max_new_tokens=100)
			generated_text = self.processor.batch_decode(generated_output[:, -100:], skip_special_tokens=True)
			for out in generated_text:
				processed_text = out.split("\x04")[-1].strip()
				answers.append(processed_text)

		return answers
