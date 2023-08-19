import os
import json
from typing import Optional, Union, List
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from hemm.data.dataset import HEMMDatasetEvaluator
from hemm.utils.common_utils import shell_command
from hemm.prompts.gqa_prompt import GQAPrompt

class GQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./'
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.prompt = GQAPrompt()

    def load(self):
      if not os.path.exists('sceneGraphs.zip'):
        shell_command('wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip')
      if not os.path.exists('questions1.2.zip'):
        shell_command('wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip')
      if not os.path.exists('images.zip'):
        shell_command('wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip')
      if not os.path.exists('gqa_images'):
        os.makedirs('gqa_images/')
        shell_command('unzip images.zip -d gqa_images')
      if not os.path.exists('gqa_scene_graphs'):
        os.makedirs('gqa_scene_graphs/')
        shell_command('unzip sceneGraphs.zip -d gqa_scene_graphs')
      if not os.path.exists('gqa_questions'):
        os.makedirs('gqa_questions/')
        shell_command('unzip questions1.2.zip -d gqa_questions')

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         metric
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric
        image_dir = 'gqa_images/images'
        annotation_file = json.load(open(os.path.join('gqa_scene_graphs', 'val_sceneGraphs.json'), 'r'))
        question_file = json.load(open(os.path.join('gqa_questions', 'val_all_questions.json'), 'r'))
        acc = []
        for data_index in tqdm(question_file, total=len(question_file)):
            # print(question_file[data_index])
            question = question_file[data_index]['question']
            image_path = os.path.join(image_dir, question_file[data_index]['imageId']+'.jpg')
            ground_truth_answer = question_file[data_index]['answer']
            text = self.get_prompt(question)
            output = self.model.generate(text, image_path)
            if output == ground_truth_answer:
                acc.append(1)
            else:
                acc.append(0)
        
        return sum(acc) / len(acc) 