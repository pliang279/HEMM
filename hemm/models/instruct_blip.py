import re
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from lavis.models import load_model_and_preprocess

class InstructBlip(HEMMModel):
	def __init__(self,
				 model_type: str,
				 ):
		super().__init__()
		self.model_type = model_type
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def load_weights(self):
		self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)

	def generate(self,
				text: Optional[str],
				image,
			) -> str:
		raw_image = Image.open(image).convert("RGB")
		image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
		generated_text = self.model.generate({"image":image, "prompt":text})
		return generated_text[0]
		
	def get_image_tensor(self, image):
		img = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
		return img

	def generate_batch(self, 
					   images,
					   texts, 
					   batch_size, 
					   ):
		answers = []
		for i in tqdm(range(0, len(texts), batch_size)):
			img_batch = images[i : i + batch_size]
			text_batch = texts[i : i + batch_size]
			generated_text = self.model.generate({"image":img_batch, "prompt":text_batch})
			answers += generated_text
				
		return answers
