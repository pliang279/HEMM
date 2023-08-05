import abc
from typing import Union
import torch


class HEMMMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self,
                ground_truth: Union[list, torch.Tensor],,
                predictions: Union[list, torch.Tensor],
                ) -> float:
        """
        Computes metric for given predictions and references.
        :param predictions: Predictions
        :param references: References
        :return:
        """
