import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.vqa_prompt import VQAPrompt

class VQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.prompt = VQAPrompt()

    def load(self):
        shell_command('wget https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Val_abstract_v002.zip')
        shell_command('wget https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Val_abstract_v002.zip')
        shell_command('wget https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip')
        shell_command('unzip scene_img_abstract_v002_val2015.zip -d ./')
        shell_command('unzip Annotations_Val_abstract_v002.zip -d ./')
        shell_command('unzip Questions_Val_abstract_v002.zip -d ./')

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text
    
    def get_ground_truth_answer(self, annotation_file, question_id):
        for data_dict in annotation_file:
            if data_dict['question_id'] == question_id:
                return data_dict['multiple_choice_answer']

    def evaluate_dataset(self,
                         model,
                         metric
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric
        image_dir = os.path.join(self.dataset_dir, 'val2014')
        annotation_file = json.load(open(os.path.join(self.dataset_dir, 'abstract_v002_val2015_annotations.json'), 'r'))
        question_file = json.load(open(os.path.join(self.dataset_dir, 'OpenEnded_abstract_v002_val2015_questions.json'), 'r'))

        acc = []
        for data_dict in tqdm(question_file['questions'], total=len(question_file)):
            question = data_dict['question']
            image_path = os.path.join(image_dir, data_dict['image_id']+'.jpg')
            ground_truth_answer = self.get_ground_truth_answer(annotation_file, data_dict['question_id'])
            text = self.get_prompt(question)
            output = self.model.generate(text, image_path)
            if output == ground_truth_answer:
                acc.append(1)
            else:
                acc.append(0)
        
        return sum(acc) / len(acc) 