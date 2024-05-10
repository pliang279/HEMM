import re
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from transformers import AutoProcessor, AutoModelForVision2Seq

def ref_text(text):
    sents = text.split("\n")
    sents = [sent.strip() for sent in sents]
    return " ".join(sents).strip()

class Kosmos2(HEMMModel):
	def __init__(self, device="cuda"):
		super().__init__()
		self.device = torch.device(device)

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
		
		prompt = "<grounding> " + text
		inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)
		generated_ids = self.model.generate(
						pixel_values=inputs["pixel_values"],
						input_ids=inputs["input_ids"][:, :-1],
						attention_mask=inputs["attention_mask"][:, :-1],
						img_features=None,
						img_attn_mask=inputs["img_attn_mask"][:, :-1],
						use_cache=True,
						max_new_tokens=500,
					)
		generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
	
		processed_text = self.processor.post_process_generation(generated_text, cleanup_and_extract=False)
		processed_text, _ = self.processor.post_process_generation(processed_text)

		processed_text = processed_text.replace(ref_text(text), "").strip()

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
			prompt_batch = ["<grounding> " + prompt for prompt in text_batch]
			inputs = self.processor(text=prompt_batch, images=img_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
			
			generated_ids = self.model.generate(
							pixel_values=inputs["pixel_values"],
							input_ids=inputs["input_ids"][:, :-1],
							attention_mask=inputs["attention_mask"][:, :-1],
							img_features=None,
							img_attn_mask=inputs["img_attn_mask"][:, :-1],
							use_cache=True,
							max_new_tokens=100,
						)
			generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
			for out, text in zip(generated_text, text_batch):
				processed_text = self.processor.post_process_generation(out, cleanup_and_extract=False)
				processed_text, _ = self.processor.post_process_generation(processed_text)
				processed_text = processed_text.replace(ref_text(text), "").strip()
				answers.append(processed_text)

		return answers
