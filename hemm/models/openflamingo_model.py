import abc
import torch
from typing import Optional, Union, List
from PIL import Image
from huggingface_hub import hf_hub_download
import torch

from hemm.models.model import HEMMModel
from hemm.utils.common_utils import shell_command

from hemm.models.open_flamingo.open_flamingo import create_model_and_transforms

class OpenFlamingoModel(HEMMModel):
    def __init__(self,
                 ) -> None:
        pass

    def load_weights(self):
        """
        Loads the model and it's processor with the initialized weight directory
        :return:
        """
        shell_command()
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.tokenizer.padding_side = "left"
    
    def get_image_tensor(self, image):
        """
        Get image tensor using model's vision processor.
        Tensor should have batch dimension.
        """
        pass

    def generate(self,
                 text: Optional[str],
                 image
                 ):
        """
        Generates output for the given prompt.
        :param text: String text prompt
        :param image: Image prompt (pillow)
        :param return_logits: return logit tensor if true, else return text/image.
        :return: return str, image or logit tensor.
        """
        image = Image.open(image)
        vision_x = [self.image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        lang_x = self.tokenizer(
            ["<image>"+str(text)],
            return_tensors="pt",
        )

        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )

        decoded_text =  self.tokenizer.decode(generated_text[0])
        return decoded_text
    
    def generate_batch(self, 
                       images: torch.Tensor,
                       texts: List[str], 
                       batch_size, 
                       ):
        """
        Batching logic for the model
        """
        pass