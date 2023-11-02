import abc
import argparse
import torch
from typing import Optional, Union, List
from PIL import Image

from huggingface_hub import snapshot_download

from hemm.models.model import HEMMModel
from hemm.models.emu.inference import prepare_model

class Emu(HEMMModel):
    def __init__(self,
                 weight_dir: str = None,
                 ) -> None:
        """
        Initializes the args for the model
        :param weight_dir: path to the weights of the model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_weights(self, model_loader_class, processor_class):
        """
        Loads the model and it's processor with the initialized weight directory
        :return:
        """
        url = snapshot_download(repo_id="BAAI/Emu", local_dir='./')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--instruct",
            action='store_true',
            default=False,
            help="Load Emu-I",
        )
        parser.add_argument(
            "--ckpt-path",
            type=str,
            default='Emu/',
            help="Emu ckpt path",
        )
        args = parser.parse_args()
        self.model = prepare_model('Emu-14B', args)
        self.model.to(self.device).to(torch.bfloat16)

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
        pass

    def generate_batch(self, 
                       images: torch.Tensor,
                       texts: List[str], 
                       batch_size, 
                       ):
        
        pass 