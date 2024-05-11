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
	def __init__(self, device="cuda"):
		self.device = torch.device(device)
		self.ckpt = 'MAGAer13/mplug-owl-llama-7b'
		self.generate_kwargs = {
			'max_new_tokens': 100,
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
	
	def reform_text(self, text):
		new_prompt = f''' The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
		Human: <image>
		Human: {text} 
		AI: '''
		
		return new_prompt

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

		inputs = self.processor(text=[self.reform_text(text)], images=[raw_image], return_tensors='pt')
		inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
		inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
		with torch.no_grad():
			res = self.model.generate(**inputs, **self.generate_kwargs)
		sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

		return sentence	
