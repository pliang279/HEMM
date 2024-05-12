import abc
from typing import Union
import torch


class HEMMMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self,
                ground_truth: Union[list, torch.Tensor],
                predictions: Union[list, torch.Tensor],
                ) -> float:
        """
        Computes metric for given predictions and references.
        :param predictions: Predictions
        :param references: References
        :return:
        """
    
    def lower(self, preds, gts):
        preds = [pred.strip().lower() for pred in preds]
        if isinstance(gts[0], str):
            gts = [gt.strip().lower() for gt in gts]
        elif isinstance(gts[0], list):
            for i, gt in enumerate(gts):
                gts[i] = [sent.strip().lower() for sent in gt]

        return preds, gts
    