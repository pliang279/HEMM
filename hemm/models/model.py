import abc
import torch
from typing import Optional, Union
from PIL import Image


class HEMMModel(abc.ABC):
    '''
    Abstract BaseModel class assuming a multi-modal foundation model.
    Takes in inputs of image and text, and outputs image/text.
    '''

    @abc.abstractmethod
    def __init__(self,
                 weight_dir: str = None,
                 ) -> None:
        """
        Initializes the args for the model
        :param weight_dir: path to the weights of the model.
        """

    @abc.abstractmethod
    def load_weights(self, model_loader_class, processor_class):
        """
        Loads the model and it's processor with the initialized weight directory
        :return:
        """

    @abc.abstractmethod
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