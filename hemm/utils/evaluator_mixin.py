import json
from typing import Optional, List

import torch
from tqdm import tqdm

from hemm.models.model import HEMMModel
from hemm.metrics.metric import HEMMMetric
from hemm.utils.base_utils import get_modality_data

class EvaluatorMixin:
    def evaluate(self,
                 model: HEMMModel,
                 loader: torch.utils.data.DataLoader,
                 output_file_path: str,
                 metrics: List[HEMMMetric],
                 modalities: Optional[List[str]]
                 ):
        results_dict = {metric.metric_name: [] for metric in metrics}
        for i, data in enumerate(tqdm(loader)):
            model_inputs = get_modality_data(data, modalities)
            outputs = model.generate(model_inputs)
            for metric in metrics:
                results = metric.compute(predictions=outputs, references=data['label'])
                results_dict[metric.metric_name].append(results)

        for metric, metric_value_list in results_dict:
            results_dict[metric] = float(sum(metric_value_list)) / len(metric_value_list)

        json.dump(results_dict, open(output_file_path, 'w'), sort_keys=True, indent=4)