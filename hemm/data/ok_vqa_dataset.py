import os
import numpy as np
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
# from hemm.data.dataset import HEMMDatasetEvaluator
# from hemm.metrics.metric import HEMMMetric
# from hemm.utils.evaluator_mixin import EvaluatorMixin

class OKVQA(Dataset):
    def __init__(self,
                 image_dir,
                 annotation_file,
                 question_file,
                 device,
                 ):
        self.image_dir = image_dir
        self.annotations = json.load(open(annotation_file, "r"))
        self.questions = json.load(open(question_file, "r"))
        self.device = device

        self.qid_to_q = {}
        for ques in self.questions["questions"]:
            self.qid_to_q[ques["question_id"]] = ques["question"]

        self.images = []
        self.qs = []
        self.gts = []

        for ann in self.annotations["annotations"]:
            self.images.append(ann["image_id"])
            self.qs.append(self.qid_to_q[ann["question_id"]])
            self.gts.append(ann)
        
    def __getitem__(self, index):
        image_id = self.images[index]
        img =  np.asarray(Image.open(os.path.join(self.image_dir, f"COCO_val2014_000000{image_id}.jpg")))
        prompt = f"Question: {self.qs[index]}"
        gt = self.gts[index]
        return {
            'image': img,
            'prompt': prompt,
            'gt': gt,
        }

    def __len__(self):
        return len(self.images)

# class OKVQAEvaluator(HEMMDatasetEvaluator, EvaluatorMixin):
#     def __init__(self,
#                  dataset_dir,
#                  model,
#                  evaluate_path,
#                  device,
#                  batch_size,
#                  shuffle_dataset,
#                  output_file_path
#                  ):
#         super().__init__(dataset_dir)
#         self.dataset_dir = dataset_dir
#         self.model = model
#         self.evaluate_path = evaluate_path
#         self.device = device
#         self.batch_size = batch_size
#         self.shuffle_dataset = shuffle_dataset
#         self.output_file_path = output_file_path

#     def evaluate_dataset(self,
#                          metrics: List[HEMMMetric],
#                          ) -> None:

#         image_dir = os.path.join(self.dataset_dir, 'val2014')        
#         annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
#         question_file = os.path.join(self.dataset_dir, 'OpenEnded_mscoco_val2014_questions.json')

#         pt_dataset = OKVQA(image_dir, annotation_file, question_file, self.device)
#         loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
#         self.evaluate(self.model, loader, self.output_file_path, modalities=['img','text'])
