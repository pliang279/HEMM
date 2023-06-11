import abc
from typing import Optional, Union

import torch
from PIL import Image

from hemm.models.model import HEMMModel


class HEMMConditionalGenerationModel(HEMMModel, abc.ABC):
    def __init__(self,
                 weight_dir: str,
                 device: str,
                 ):
        super().__init__(weight_dir)
        self.base_model_key = weight_dir
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self, model_loader_class, processor_class):
        self.model = model_loader_class.from_pretrained(self.base_model_key)
        self.processor = processor_class.from_pretrained(self.base_model_key)
        self.model.to(self.device)

    def generate(self,
                 text: Optional[str],
                 image: Optional[Image],
                 ) -> str:
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
