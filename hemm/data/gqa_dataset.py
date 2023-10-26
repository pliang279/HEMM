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
from hemm.metrics.bertscore_metric import BertScoreMetric
from hemm.metrics.bleu_metric import BleuMetric

class GQADatasetEvaluator(HEMMDatasetEvaluator):
    def __init__(self,
                 dataset_dir='./'
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.prompt = GQAPrompt()
        self.metrics = [BertScoreMetric(), BleuMetric()]

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
                         ) -> None:
        self.load()
        self.model = model
        image_dir = 'gqa_images/images'
        annotation_file = json.load(open(os.path.join('gqa_scene_graphs', 'val_sceneGraphs.json'), 'r'))
        question_file = json.load(open(os.path.join('gqa_questions', 'val_all_questions.json'), 'r'))
        
        ground_truth = []
        predictions = []
        for data_index in tqdm(question_file, total=len(question_file)):
            # print(question_file[data_index])
            question = question_file[data_index]['question']
            image_path = os.path.join(image_dir, question_file[data_index]['imageId']+'.jpg')
            ground_truth_answer = question_file[data_index]['answer']
            text = self.get_prompt(question)
            output = self.model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)
        
        results = {}
        for metric in self.metrics:
           results[metric.name] = metric.compute(ground_truth, predictions)
        return results

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.load()
        self.model = model
        image_dir = 'gqa_images/images'
        annotation_file = json.load(open(os.path.join('gqa_scene_graphs', 'val_sceneGraphs.json'), 'r'))
        question_file = json.load(open(os.path.join('gqa_questions', 'val_all_questions.json'), 'r'))
        
        ground_truth = []
        predictions = []

        texts = []
        images = []

        for data_index in tqdm(question_file, total=len(question_file)):
            # print(question_file[data_index])
            question = question_file[data_index]['question']
            image_path = os.path.join(image_dir, question_file[data_index]['imageId']+'.jpg')
            raw_image = Image.open(image_path).convert('RGB')
            image = self.model.get_image_tensor(raw_image)
            images.append(image)

            ground_truth_answer = question_file[data_index]['answer']
            text = self.get_prompt(question)
            texts.append(text)
            ground_truth.append(ground_truth_answer)
        
        images_tensor = torch.cat(images, dim=0)
        images_tensor = images_tensor.to(self.model.device)
        predictions = self.model.generate_batch(images_tensor, texts, batch_size)

        results = {}
        for metric in self.metrics:
           results[metric.name] = metric.compute(ground_truth, predictions)
           
        return results
    