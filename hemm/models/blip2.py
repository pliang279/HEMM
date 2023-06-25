from typing import Optional, Union

import torch
from PIL import Image

from hemm.models.model import HEMMModel
from lavis.models import load_model_and_preprocess

class BLIP2(HEMMModel):
    def __init__(self,
                 model_type: str,
                 device: str,
                 ):
        super().__init__(weight_dir)
        self.model_type = model_type
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        self.model, self.processor, _ = load_model_and_preprocess(
            name="blip2_t5", model_type=self.model_type, is_eval=True, device=self.device)

    def generate(self,
                 text: Optional[str],
                 image: Optional[Image],
                 ) -> str:

        processed_image = self.processor["eval"](image).unsqueeze(0).to(self.device)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        generated_text = model.generate({"image": image, "prompt":text})[0].strip()
        return generated_text
        
        