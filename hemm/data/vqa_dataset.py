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
import ast

class VQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./'
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.prompt = VQAPrompt()

    def load(self):
      if not os.path.exists('Annotations_Val_abstract_v002.zip'):
        shell_command('wget https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Val_abstract_v002.zip')
      if not os.path.exists('Questions_Val_abstract_v002.zip'):
        shell_command('wget https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Val_abstract_v002.zip')
      if not os.path.exists('scene_img_abstract_v002_val2015.zip'):
        shell_command('wget https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip')
      if not os.path.exists('vqa_images'):
        os.makedirs('vqa_images/')
        shell_command('unzip scene_img_abstract_v002_val2015.zip -d vqa_images')
      if not os.path.exists('vqa_annotations'):
        os.makedirs('vqa_annotations/')
        shell_command('unzip Annotations_Val_abstract_v002.zip -d vqa_annotations')
      if not os.path.exists('vqa_questions'):
        os.makedirs('vqa_questions/')
        shell_command('unzip Questions_Val_abstract_v002.zip -d vqa_questions')

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text
    
    def get_ground_truth_answer(self, annotation_file, question_id):
        for data_dict in annotation_file['annotations']:
            if data_dict['question_id'] == question_id:
                return data_dict['multiple_choice_answer']

    def evaluate_dataset(self,
                         model,
                         metric
                         ) -> None:
        self.load()
        self.model = model
        self.metric = metric
        image_dir = 'vqa_images'
        annotation_file = json.load(open(os.path.join('vqa_annotations', 'abstract_v002_val2015_annotations.json'), 'r'))
        question_file = json.load(open(os.path.join('vqa_questions', 'OpenEnded_abstract_v002_val2015_questions.json'), 'r'))
        
        ground_truth = []
        predictions = []
        for data_dict in tqdm(question_file['questions'], total=len(question_file['questions'])):
            question = data_dict['question']
            image_path = os.path.join(image_dir, "abstract_v002_val2015_0000000{}".format(str(data_dict['image_id']))+'.png')
            ground_truth_answer = self.get_ground_truth_answer(annotation_file, data_dict['question_id'])
            text = self.get_prompt(question)
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
        
        results = self.metric.compute(ground_truth, predictions)
        return results
    
    def evaluate_dataset_batched(self,
                                 model,
                                metric,
                                batch_size = 32
                                ):
      self.load()
      self.metric = metric
      self.model = model
      texts = []
      images = []
      ground_truth = []
      image_dir = 'vqa_images'
      annotation_file = json.load(open(os.path.join('vqa_annotations', 'abstract_v002_val2015_annotations.json'), 'r'))
      question_file = json.load(open(os.path.join('vqa_questions', 'OpenEnded_abstract_v002_val2015_questions.json'), 'r'))

      for data_dict in tqdm(question_file['questions'], total=len(question_file['questions'])):
          question = data_dict['question']
          image_path = os.path.join(image_dir, "abstract_v002_val2015_0000000{}".format(str(data_dict['image_id']))+'.png')
          raw_image = Image.open(image_path).convert('RGB')
          image = self.model.get_image_tensor(raw_image)
          images.append(image)
          ground_truth_answer = self.get_ground_truth_answer(annotation_file, data_dict['question_id'])
          text = self.get_prompt(question)
          texts.append(text)
          ground_truth.append(ground_truth_answer)

      images_tensor = torch.cat(images, dim=0)
      images_tensor = images_tensor.to(self.model.chat.device)
      predictions = self.model.generate_batch(images_tensor, texts, batch_size)

      results = self.metric.compute(ground_truth, predictions)
      return results