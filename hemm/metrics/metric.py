import abc
from typing import Union
import torch


class HEMMMetric(abc.ABC):
    @abc.abstractmethod
    def __init__(self,
                 metric_name: str,
                 ):
        """
        Initializes metric.
        :param metric_name: Name of metric to be used.
        """
        self.metric_name = metric_name

    @abc.abstractmethod
    def compute(self,
                predictions: Union[list, torch.Tensor],
                references: Union[list, torch.Tensor],
                ) -> float:
        """
        Computes metric for given predictions and references.
        :param predictions: Predictions
        :param references: References
        :return:
        """
