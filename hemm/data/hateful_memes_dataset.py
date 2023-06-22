import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch

from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.metrics.metric import HEMMMetric
from hemm.utils.evaluator_mixin import EvaluatorMixin
from hemm.prompts.hateful_memes_prompt import HatefulMemesPrompt


class HatefulMemesPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_dir,
                 label_json_file,
                 image_processor,
                 text_processor,
                 prompt,
                 device,
                 ):
        self.image_dir = image_dir
        self.labels = label_json_file
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.prompt = prompt
        self.device = device

    def __getitem__(self, index):
        json_obj = dict(self.labels[index])
        img = self.image_processor(os.path.join(self.image_dir, json_obj['img']))
        label = torch.tensor(self.labels['label'], device=self.device)
        prompt_text = self.get_prompt(self.labels['text'])
        text = self.text_processor(prompt_text)
        return {
            'image': img,
            'text': text,
            'label': label
        }

    def get_prompt(self, text) -> str:
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

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

        self.dataset_dir = dataset_dir
        self.model = model
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.evaluate_path = evaluate_path
        self.device = device
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.output_file_path = output_file_path
        self.prompt = HatefulMemesPrompt()

    def evaluate_dataset(self,
                         metrics: List[HEMMMetric],
                         ) -> None:
        label_path = os.path.join(self.dataset_dir, 'data', self.evaluate_path)
        json_list = list(open(label_path, 'r'))
        image_dir = os.path.join(self.dataset_dir, 'data', 'img')
        pt_dataset = HatefulMemesPTDataset(image_dir, json_list, self.image_processor, self.text_processor, self.prompt, self.device)
        loader = torch.utils.data.DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        self.evaluate(self.model, loader, self.output_file_path, modalities=['img','text'])