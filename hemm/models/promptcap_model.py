from typing import Optional, Union

import torch
from promptcap import PromptCap_VQA
from PIL import Image
import re
from hemm.models.model import HEMMModel
from hemm.utils.common_utils import shell_command

class PromptCap(HEMMModel):
    def __init__(self):
        super().__init__()


    def load_weights(self):
        self.vqa_model = PromptCap_VQA(promptcap_model="vqascore/promptcap-coco-vqa", qa_model="allenai/unifiedqa-t5-base")
        
        if torch.cuda.is_available():
            self.vqa_model.cuda()

    def generate(self,
                text: Optional[str],
                image,
            ) -> str:        
        # upload image
        image_name = f"promptcap_im_input.jpg"
        image.save(image_name)

        # get answer
        llm_message = self.vqa_model.vqa(text, image_name)
        shell_command('rm promptcap_im_input.jpg')
        return llm_message

    def answer_extractor(self, text, dataset_key):
        if dataset_key == 'hateful_memes' or dataset_key =='newyorkercartoon' or dataset_key =='irfl':
            text = text[:3]
            text = text.lower().strip()
            text = ''.join(filter(str.isalpha, text.lower()))
            return text
        elif dataset_key == 'memotion' or dataset_key == 'face_emotion' or dataset_key == 'scienceqa' or dataset_key == 'vcr':
            match = re.search(r"\b\d\b", text)
            if match:
                first_number = int(match.group())
                return first_number
            else:
                return None
        elif dataset_key == 'winogroundvqa':
            answer = ''.join(filter(str.isalpha, text.split()[0].lower()))
            return answer
    
    def get_image_tensor(self, image):
        pass 

    def generate_batch(self, 
                       images,
                       texts, 
                       batch_size, 
                       ):
        pass
