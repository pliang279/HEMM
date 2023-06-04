import abc
import torch
from typing import Optional, Union, List

from model import Model
from metric import Metric


class Dataset(abc.ABC):
    """
    General dataset class used for evaluating a Model on a set of metrics.
    """
    @abc.abstractmethod
    def __init__(self,
                 dataset_path: str,
                 ):
        """
        Initialize dataset
        :param dataset_path: path to downloaded dataset
        """

    @abc.abstractmethod
    def evaluate(self,
                 model: Model,
                 metrics: List[Metric],
                 ):
        """
        :param model: model which can evaluate on the whole dataset.
        :param metrics: list of metrics used for evaluation.
        :return:
        """
