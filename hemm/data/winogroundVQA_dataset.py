import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric

class WinogroundVQA(Dataset):
    def __init__(self,
                 image_dir,
                 questions_file,
                 device,
                 ):
        self.image_dir = image_dir
        self.questions = load_dataset("csv", data_files=questions_file)['train']
        self.device = device
        
    def __getitem__(self, index):
        res = []
        for pair in ((0,0), (0,1), (1,0), (1,1)):
            im_k = pair[0]
            qu_k = pair[1]
            img = Image.open(os.path.join(self.image_dir, f"ex_{index}_img_{im_k}.png"))
            question = self.questions[index][f'query_{qu_k}']
            gt = 'yes' if pair[0] == pair[1] else 'no'
            res.append({
                'image': img,
                'question': question,
                'gt': gt
            })
        return res

    def __len__(self):
        return len(self.images)

class WinogroundVQAEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir,
                 model,
                 evaluate_path,
                 device,
                 batch_size,
                 shuffle_dataset,
                 output_file_path
                 ):
        super().__init__(dataset_dir)
        self.dataset_dir = dataset_dir
        self.model = model
        self.evaluate_path = evaluate_path
        self.device = device
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.output_file_path = output_file_path

    def evaluate(self,
                         metrics: List[HEMMMetric],
                         ) -> None:

        image_dir = os.path.join(self.dataset_dir, 'images')
        questions_file = os.path.join(self.dataset_dir, 'winoground_vqa.csv')

        pt_dataset = WinogroundVQA(image_dir, questions_file, self.device)
        loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)

        # This is the concrete implementation of the `evaluate` function.
        self._evaluate(self.model, loader, self.output_file_path, modalities=['img','text'])

    def _evaluate(self,
                         model,
                         loader,
                         output_file_path,
                         modalities=['img','text']) -> None:
        pass
