import abc
import torch
from typing import Optional, Union, List

from hemm.models.model import HEMMModel
from hemm.metrics.metric import HEMMMetric


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
                         metrics: List[HEMMMetric],
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