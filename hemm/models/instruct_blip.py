import re
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from lavis.models import load_model_and_preprocess


class InstructBlip(HEMMModel):
	def __init__(self, device="cuda", **kwargs):
		super().__init__()
		self.model_type = kwargs["model_type"]
		self.device = torch.device(device)

	def load_weights(self):
		self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", 
																 		model_type=self.model_type, 
																		is_eval=True, 
																		device=self.device)

	def generate(self,
				text: Optional[str],
				image,
			) -> str:
		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image
		
		image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
		generated_text = self.model.generate({"image":image, "prompt":text})
		return generated_text[0]
		
	def get_image_tensor(self, image):
		img = self.vis_processors["eval"](image).unsqueeze(0)
		return img
	
	def answer_extractor(self, text, dataset_key):
		return text

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
