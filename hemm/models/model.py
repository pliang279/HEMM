import abc
import torch
from typing import Optional, Union, List
from PIL import Image


class HEMMModel(abc.ABC):
	'''
	Abstract BaseModel class assuming a multi-modal foundation model.
	Takes in inputs of image and text, and outputs image/text.
	'''

	@abc.abstractmethod
	def __init__(self,
				 weight_dir: str = None,
				 ) -> None:
		"""
		Initializes the args for the model
		:param weight_dir: path to the weights of the model.
		"""

	@abc.abstractmethod
	def load_weights(self, model_loader_class, processor_class):
		"""
		Loads the model and it's processor with the initialized weight directory
		:return:
		"""

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
	
	@abc.abstractmethod
	def get_image_tensor(self, image):
		"""
		Get image tensor using model's vision processor.
		Tensor should have batch dimension.
		"""

	@abc.abstractmethod
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
	
	@abc.abstractmethod
	def generate_batch(self, 
					   images: torch.Tensor,
					   texts: List[str], 
					   batch_size, 
					   ):
		"""
		Batching logic for the model
		"""