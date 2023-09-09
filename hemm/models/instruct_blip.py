import re
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

class InstructBlip(HEMMModel):
	def __init__(self,
				 model_type: str,
				 ):
		super().__init__()
		self.model_type = model_type
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def load_weights(self):
		self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").to(self.device)
		self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b").to(self.device)

	def generate(self,
				text: Optional[str],
				image,
			) -> str:
		image = Image.open(image).convert("RGB")
		inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
		outputs = self.model.generate(**inputs,
								do_sample=False,
								num_beams=5,
								max_length=256,
								min_length=1,
								top_p=0.9,
								repetition_penalty=1.5,
								length_penalty=1.0,
								temperature=1,
								)
		generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
		return generated_text
		
	def get_image_tensor(self, image):
		img = self.processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(self.device)

	def generate_batch(self, 
					   images,
					   texts, 
					   batch_size, 
					   ):
		answers = []
		for i in tqdm(range(0, len(texts), batch_size)):
			img_batch = images[i : i + batch_size]
			text_batch = texts[i : i + batch_size]
			text_inputs = self.processor(text=text_batch, return_tensors="pt", padding="max_length", max_length=256, truncation=True).to(self.device)
			outputs = self.model.generate(**text_inputs,
									pixel_values=img_batch,
									do_sample=False,
									num_beams=5,
									max_length=256,
									min_length=1,
									top_p=0.9,
									repetition_penalty=1.5,
									length_penalty=1.0,
									temperature=1)
			
			generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
			answers += generated_text
		
		return answers
