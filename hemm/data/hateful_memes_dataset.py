import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.utils.evaluator_mixin import EvaluatorMixin


class HatefulMemesPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_dir,
                 label_json_file,
                 image_processor,
                 text_processor,
                 device,
                 ):
        self.image_dir = image_dir
        self.labels = label_json_file
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.device = device

    def __getitem__(self, index):
        json_obj = dict(self.labels[index])
        img = self.image_processor(os.path.join(self.image_dir, json_obj['img']))
        label = torch.tensor(self.labels['label'], device=self.device)
        text = self.text_processor(self.labels['text'])
        return {
            'image': img,
            'text': text,
            'label': label
        }

    def __len__(self):
        return len(self.labels)


class HatefulMemesDatasetEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
    def __init__(self,
                 dataset_dir,
                 model,
                 text_processor,
                 image_processor,
                 evaluate_path,
                 device,
                 batch_size,
                 shuffle_dataset,
                 output_file_path
                 ):
        super().__init__(dataset_dir)
        """
        Class for hateful memes dataset. Assuming data is already downloaded at dataset_path directory.
        """
        self.dataset_dir = dataset_dir
        self.model = model
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.evaluate_path = evaluate_path
        self.device = device
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.output_file_path = output_file_path

    def evaluate_dataset(self,
                         metrics: List[HEMMMetric],
                         ) -> None:
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data', 'img')
        pt_dataset = HatefulMemesPTDataset(image_dir, json_list, self.image_processor, self.text_processor, self.device)
        loader = torch.utils.data.DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        self.evaluate(self.model, loader, self.output_file_path, modalities=['img','text'])