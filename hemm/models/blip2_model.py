from typing import Optional, Union

import torch
from PIL import Image
import re
from hemm.models.model import HEMMModel
from lavis.models import load_model_and_preprocess

class BLIP2(HEMMModel):
    def __init__(self,
                 model_type: str,
                 ):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_weights(self):
        self.model, self.processor, _ = load_model_and_preprocess(
            name="blip2_t5", model_type=self.model_type, is_eval=True, device=self.device)

    def generate(self,
                text: Optional[str],
                image,
            ) -> str:
        image = Image.open(image).convert("RGB")
        processed_image = self.processor["eval"](image).unsqueeze(0).to(self.device)
        generated_text = self.model.generate({"image": processed_image, "prompt":text})[0].strip()
        return generated_text

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
    
    def get_image_tensor(self, image):
        pass 

    def generate_batch(self, 
                       images,
                       texts, 
                       batch_size, 
                       ):
        pass