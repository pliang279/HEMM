import abc
import argparse
import torch
import json
from typing import Optional, Union, List
from tqdm import tqdm
from PIL import Image

from hemm.models.model import HEMMModel
from hemm.models.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from hemm.models.mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from hemm.models.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

class MplugOwl(HEMMModel):
	def __init__(self,
				 pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b',
				 ) -> None:
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.ckpt = pretrained_ckpt
		self.generate_kwargs = {
			'do_sample': True,
			'top_k': 5,
			'max_length': 512
			}

	def load_weights(self):
		self.model = MplugOwlForConditionalGeneration.from_pretrained(self.ckpt, torch_dtype=torch.bfloat16).to(self.device)
		self.image_processor = MplugOwlImageProcessor.from_pretrained(self.ckpt, device=self.device)
		self.tokenizer = MplugOwlTokenizer.from_pretrained(self.ckpt, device=self.device)
		self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)

	def get_image_tensor(self, image):
		"""
		Get image tensor using model's vision processor.
		Tensor should have batch dimension.
		"""
		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image
		
		return raw_image

	def generate(self,
				 text: Optional[str],
				 image
				 ):
		"""
		Generates output for the given prompt.
		:param text: String text prompt
		:param image: Image prompt (pillow)
		:param return_logits: return logit tensor if true, else return text/image.
		:return: return str, image or logit tensor.
		"""
		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image

		inputs = self.processor(text=[text], images=[raw_image], return_tensors='pt')
		inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
		inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
		with torch.no_grad():
			res = self.model.generate(**inputs, **self.generate_kwargs)
		sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

		return sentence	

	def generate_batch(self, 
					   images: torch.Tensor,
					   texts: List[str], 
					   batch_size, 
					   ):
		
		answers = []
		for i in tqdm(range(0, len(texts), batch_size)):
			img_batch = images[i : i + batch_size]
			text_batch = texts[i : i + batch_size]

			inputs = self.processor(text=text_batch, images=img_batch, return_tensors='pt')
			inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
			inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
			with torch.no_grad():
				res = self.model.generate(**inputs, **self.generate_kwargs)
			
			for out in res.tolist():
				sentence = self.tokenizer.decode(out, skip_special_tokens=True)
				answers.append(sentence)
		
		return answers
