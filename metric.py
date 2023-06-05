import abc
from typing import Union
import torch


class Metric(abc.ABC):
    @abc.abstractmethod
    def __init__(self,
                 metric_name: str,
                 ):
        """
        Initializes metric.
        :param metric_name: Name of metric to be used.
        """

    @abc.abstractmethod
    def compute(self,
                predictions: Union[list, torch.Tensor],
                references: Union[list, torch.Tensor],
                ):
        """
        Computes metric for given predictions and references.
        :param predictions: Predictions
        :param references: References
        :return:
        """
