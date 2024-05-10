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
                  download_dir="./",
                  dataset_dir=None,
                  annotation_file=None,
                  **kwargs,
                  ):
        super().__init__()
        self.dataset_dir = download_dir 
        self.prompt = GQAPrompt()
        self.load()

    def __len__(self):       
       question_file = json.load(open(os.path.join(f'{self.dataset_dir}/gqa_questions', 'testdev_all_questions.json'), 'r'))
       return len(question_file)

    def load(self):
      if not os.path.exists(f'{self.dataset_dir}/sceneGraphs.zip'):
        shell_command(f'wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip -d {self.dataset_dir}')
      if not os.path.exists(f'{self.dataset_dir}/questions1.2.zip'):
        shell_command(f'wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -d {self.dataset_dir}')
      if not os.path.exists(f'{self.dataset_dir}/images.zip'):
        shell_command(f'wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -d {self.dataset_dir}')
      if not os.path.exists(f'{self.dataset_dir}/gqa_images'):
        os.makedirs(f'{self.dataset_dir}/gqa_images/')
        shell_command(f'unzip {self.dataset_dir}/images.zip -d {self.dataset_dir}/gqa_images')
      if not os.path.exists(f'{self.dataset_dir}/gqa_scene_graphs'):
        os.makedirs(f'{self.dataset_dir}/gqa_scene_graphs/')
        shell_command(f'unzip {self.dataset_dir}/sceneGraphs.zip -d {self.dataset_dir}/gqa_scene_graphs')
      if not os.path.exists(f'{self.dataset_dir}/gqa_questions'):
        os.makedirs(f'{self.dataset_dir}/gqa_questions/')
        shell_command(f'unzip {self.dataset_dir}/questions1.2.zip -d {self.dataset_dir}/gqa_questions')

    def get_prompt(self, text):
        prompt_text = self.prompt.format_prompt(text)
        return prompt_text

    def evaluate_dataset(self,
                         model,
                         ) -> None:
        image_dir = f'{self.dataset_dir}/gqa_images/'
        question_file = json.load(open(os.path.join(f'{self.dataset_dir}/gqa_questions', 'testdev_all_questions.json'), 'r'))
        
        ground_truth = []
        predictions = []
        for data_index in tqdm(question_file, total=len(question_file)):
            question = question_file[data_index]['question']
            image_path = os.path.join(image_dir, question_file[data_index]['imageId']+'.jpg')
            ground_truth_answer = question_file[data_index]['answer']
            text = self.get_prompt(question)
            output = model.generate(text, image_path)
            predictions.append(output)
            ground_truth.append(ground_truth_answer)

        return predictions, ground_truth

    def evaluate_dataset_batched(self,
                         model,
                         batch_size=32
                         ) -> None:
        self.model = model
        image_dir = f'{self.dataset_dir}/gqa_images/'

        question_file = json.load(open(os.path.join(f'{self.dataset_dir}/gqa_questions', 'testdev_all_questions.json'), 'r'))
        
        ground_truth = []
        predictions = []

        texts = []
        images = []
        all_preds = []
        for data_index in tqdm(question_file, total=len(question_file)):
            question = question_file[data_index]['question']
            image_path = os.path.join(image_dir, question_file[data_index]['imageId']+'.jpg')
            image = self.model.get_image_tensor(Image.open(image_path).convert('RGB'))
            images.append(image)

            ground_truth_answer = question_file[data_index]['answer']
            text = self.get_prompt(question)
            texts.append(text)
            ground_truth.append(ground_truth_answer)

            if len(images) % batch_size == 0:
              predictions = self.predict_batched(images, texts, batch_size)
              all_preds += predictions
              images = []
              texts = []
            
        if len(images) > 0:
          predictions = self.predict_batched(images, texts, batch_size)
          all_preds += predictions
    
        return all_preds, ground_truth
    