from typing import Optional, Union
import torch
from PIL import Image
import re
from hemm.models.model import HEMMModel
from tqdm import tqdm
from hemm.models.gill import models

class GILL(HEMMModel):
	def __init__(self,
				 model_dir="/home/agoindan/gill/checkpoints/gill_opt/"
				 ):
		super().__init__()
		self.model_dir = model_dir
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def load_weights(self):
		self.model = models.load_gill(self.model_dir)

	def generate_image(self,
				text: Optional[str],
				image,
				):
		image = Image.open(image).convert("RGB")
		prompts = [image, text]
		g_cuda = torch.Generator(device=self.device)
		outputs = self.model.generate_for_images_and_texts(prompts, 
													 num_words=2,
													 ret_scale_factor=100.0,
													 generator=g_cuda)
		if outputs[1]['decision'][0] == 'gen':
			return outputs[1]['gen'][0][0]
		
		return outputs[1]['ret'][0][0]

	def generate(self, 
					text: str,
					image):
		image = Image.open(image).convert("RGB")
		prompts = [image, text]
		outputs = self.model.generate_for_images_and_texts(prompts, num_words=100, min_word_tokens=100, top_p=0.95, temperature=0.6)

		generated_text = outputs[0]
		return generated_text

	def answer_extractor(self, text, dataset_key):
		if dataset_key == 'hateful_memes' or dataset_key =='newyorkercartoon' or dataset_key =='irfl':
			text = text[:3]
			text = text.lower().strip()
			text = ''.join(filter(str.isalpha, text.lower()))
			return text
		elif dataset_key == 'memotion' or dataset_key == 'face_emotion' or dataset_key == 'scienceqa' or dataset_key == 'vcr':
			match = re.search(r"\b\d\b", text)
			if match:
				first_number = int(match.group())
				return first_number
			else:
				return None
	
	def get_image_tensor(self, image):
		pass

	def generate_batch(self, 
					   images,
					   texts, 
					   batch_size, 
					   ):
		pass
	