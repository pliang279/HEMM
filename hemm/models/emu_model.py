import abc
import argparse
import torch
from typing import Optional, Union, List
from PIL import Image

from huggingface_hub import snapshot_download

from hemm.models.model import HEMMModel
from hemm.models.emu.inference import prepare_model
from hemm.models.emu.models.pipeline import EmuGenerationPipeline
from hemm.models.emu.utils import process_img
from hemm.models.emu.inference import Emu_inference

class Emu(HEMMModel):
    def __init__(self,
                 weight_dir: str = None,
                 ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
        self.image_system_msg = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."

    def load_weights(self, model_loader_class, processor_class):
        url = snapshot_download(repo_id="BAAI/Emu", local_dir='./')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--instruct",
            action='store_true',
            default=True,
            help="Load Emu-I",
        )
        parser.add_argument(
            "--ckpt-path",
            type=str,
            default='Emu/Emu-instruct.pt',
            help="Emu ckpt path",
        )
        args = parser.parse_args()
        self.model = prepare_model('Emu-14B', args)
        self.model.to(self.device).to(torch.bfloat16)

        parser2 = argparse.ArgumentParser()
        parser2.add_argument(
            "--instruct",
            action='store_true',
            default=False,
            help="Load Emu-I",
        )
        parser2.add_argument(
            "--ckpt-path",
            type=str,
            default='Emu/pretrain',
            help="Emu Decoder ckpt path",
        )
        self.image_generation_args = parser2.parse_args()

        self.image_generation_pipeline = EmuGenerationPipeline.from_pretrained(
            path=self.image_generation_args.ckpt_path,
            args=self.image_generation_args,
        )
        self.image_generation_pipeline = self.image_generation_pipeline.bfloat16().to(self.device)

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
        image = process_img(img_path='examples/iron_man.jpg', device=self.device)
        output = Emu_inference([image], self.image_placeholder + text, system=self.image_system_msg)
        return output

    def generate_batch(self, 
                       images: torch.Tensor,
                       texts: List[str], 
                       batch_size, 
                       ):
        
        pass

    def generate_image(self,
                       text: Optional[str],
				       image,
                       ):
        image, safety = self.image_generation_pipeline(
            [text],
            height=512,
            width=512,
            guidance_scale=7.5,
        )
        return image