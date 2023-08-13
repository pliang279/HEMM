import os
import numpy as np
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.okqvqa_prompt import OKVQAPrompt

class OKVQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir,
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = 'okvqa'
        self.prompt = OKVQAPrompt()

    def load(self):
        shell_command('wget http://images.cocodataset.org/zips/val2014.zip')
        shell_command('wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip')
        shell_command('wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip')
        shell_command('unzip mscoco_val2014_annotations.json.zip -d ./')
        shell_command('unzip OpenEnded_mscoco_val2014_questions.json.zip -d ./')
        shell_command('unzip val2014.zip -d ./')

    def get_prompt(self, question):
        self.prompt.format_prompt(question)
        return self.prompt

    def evaluate_dataset(self,
                         model,
                         metric,
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric

        image_dir = os.path.join(self.dataset_dir, 'val2014')        
        annotation_file = os.path.join(self.dataset_dir, 'mscoco_val2014_annotations.json')
        question_file = os.path.join(self.dataset_dir, 'OpenEnded_mscoco_val2014_questions.json')

        annotations = json.load(open(annotation_file, "r"))
        questions = json.load(open(question_file, "r"))

        qid_to_q = {}
        for ques in questions["questions"]:
            qid_to_q[ques["question_id"]] = ques["question"]
        
        images = []
        qs = []
        ground_truth = []

        for ann in annotations["annotations"]:
            images.append(ann["image_id"])
            qs.append(qid_to_q[ann["question_id"]])
            ground_truth.append(ann)
        
        acc = []
        for i in tqdm(range(len(images)), total=len(images)):
            image_path = os.path.join(image_dir, f"COCO_val2014_000000{images[i]}.jpg")
            text = self.get_prompt(qs[i])
            output = self.model.generate(text, image_path)
            ground_truth_answer = ground_truth[i]['answers'][0]['raw_answer']
            if output == ground_truth_answer:
                acc.append(1)
            else:
                acc.append(0)
        
        return sum(acc)/len(acc)