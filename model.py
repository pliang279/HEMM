import abc
import torch
from typing import Optional, Union
from PIL import Image


class Model(abc.ABC):
    '''
    Abstract BaseModel class assuming a multi-modal foundation model.
    Takes in inputs of image and text, and outputs image/text.
    '''

    @abc.abstractmethod
    def __init__(self,
                 weight_dir: str,
                 ) -> None:
        """
        Initializes the model
        :param weight_dir: path to the weights of the model.
        """

    @abc.abstractmethod
    def generate(self,
                 text: Optional[str],
                 image: Optional[Image],
                 return_logits: Optional[bool]
                 ) -> Union[str, Image, torch.FloatTensor]:
        """
        Generates output for the given prompt.
        :param text: String text prompt
        :param image: Image prompt (pillow)
        :param return_logits: return logit tensor if true, else return text/image.
        :return: return str, image or logit tensor.
        """