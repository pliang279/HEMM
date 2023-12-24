import abc
import torch
from typing import Optional, Union, List

from hemm.models.model import HEMMModel
from hemm.metrics.metric import HEMMMetric
from PIL import Image
import pickle

class HEMMDatasetEvaluator(abc.ABC):
	"""
	General dataset class used for evaluating a Model on a set of metrics. This class will be used for pre-processing
	and handling of data.
	"""

	@abc.abstractmethod
	def __init__(self,
				 dataset_dir: str = None,
				 ):
		"""
		Initialize dataset
		:param dataset_path: path to downloaded dataset
		"""
	
	@abc.abstractmethod
	def evaluate_dataset(self,
						 ):
		"""
		:param model: model which can evaluate on the whole dataset.
		:param metrics: list of metrics used for evaluation.
		:return:
		"""
	
	@abc.abstractmethod
	def load(self):
		"""
		download dataset script
		"""

	def predict_batched(self, images, texts, batch_size):
		"""
		make predictions for batched inference
		"""
		if isinstance(images[0], Image.Image):
			predictions = self.model.generate_batch(images, texts, batch_size)
		else:
			images_tensor = torch.cat(images, dim=0)
			images_tensor = images_tensor.to(self.model.device)
			predictions = self.model.generate_batch(images_tensor, texts, batch_size)

		return predictions
	
	def save_details(self, images, texts, gts, name):
		assert len(images) == len(texts) == len(gts)
		details = {
			"images":images,
			"texts": texts,
			"gts": gts
		}
		pickle.dump(details, open(name, "wb"))

		return

	def run_metrics(self, ground_truth, predictions):
		results = {}
		for metric in self.metrics:
			results[metric.name] = metric.compute(ground_truth, predictions)
			
		return results

	@abc.abstractmethod
	def evaluate_dataset_batched(self):
		"""
		Evaluate dataset in a batched format
		"""    