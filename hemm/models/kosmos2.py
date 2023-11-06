import re
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from transformers import AutoProcessor, AutoModelForVision2Seq

class Kosmos2(HEMMModel):
	def __init__(self,
				 ):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def load_weights(self):
		self.model = AutoModelForVision2Seq.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True).to(self.device)
		self.processor = AutoProcessor.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)

	def generate(self,
				text: Optional[str],
				image,
			) -> str:
		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image
		
		inputs = self.processor(text=text, images=raw_image, return_tensors="pt").to(self.device)
		generated_ids = self.model.generate(
						pixel_values=inputs["pixel_values"],
						input_ids=inputs["input_ids"][:, :-1],
						attention_mask=inputs["attention_mask"][:, :-1],
						img_features=None,
						img_attn_mask=inputs["img_attn_mask"][:, :-1],
						use_cache=True,
						max_new_tokens=64,
					)
		generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
	
		processed_text = self.processor.post_process_generation(generated_text, cleanup_and_extract=False)
		processed_text, _ = self.processor.post_process_generation(processed_text)

		return processed_text
			
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
			inputs = self.processor(text=text_batch, images=img_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
			generated_ids = self.model.generate(
							pixel_values=inputs["pixel_values"],
							input_ids=inputs["input_ids"][:, :-1],
							attention_mask=inputs["attention_mask"][:, :-1],
							img_features=None,
							img_attn_mask=inputs["img_attn_mask"][:, :-1],
							use_cache=True,
							max_new_tokens=64,
						)
			generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
			for out in generated_text:
				processed_text = self.processor.post_process_generation(out, cleanup_and_extract=False)
				processed_text, _ = self.processor.post_process_generation(processed_text)
				answers.append(processed_text)

		return answers
