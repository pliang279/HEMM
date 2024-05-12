import os
import abc
import argparse
import torch
import json
from typing import Optional, Union, List
from PIL import Image

from huggingface_hub import snapshot_download

from hemm.models.model import HEMMModel
from hemm.models.emu.models.modeling_emu import Emu
from hemm.models.emu.utils import process_img
from hemm.utils.common_utils import shell_command
from tqdm import tqdm

def parse_args(download_dir):
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
		default=f'{download_dir}/Emu-pretrain.pt',
		help="Emu ckpt path",
	)
	args = parser.parse_args('')

	return args

class EmuModel(HEMMModel):
	def __init__(self, device="cuda", download_dir="./", **kwargs):
		self.device = device
		self.download_dir = f"{download_dir}/emu/"
		self.image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
		self.system = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."
		self.args = parse_args(self.download_dir)
		self.args.device = torch.device(self.device)
		current_dir = os.path.dirname(os.path.abspath(__file__))
		cfg_file = os.path.join(current_dir, "emu/models/Emu-14B.json")
		with open(cfg_file, "r", encoding="utf8") as f:
			self.model_cfg = json.load(f)

	def load_weights(self):
		if not os.path.exists(self.download_dir):
			shell_command(f"mkdir -p {self.download_dir}")
			shell_command(f"wget https://huggingface.co/BAAI/Emu/resolve/main/Emu-pretrain.pt -P {self.download_dir}")

		self.model = Emu(**self.model_cfg, cast_dtype=torch.float, args=self.args)
		ckpt = torch.load(self.args.ckpt_path, map_location="cpu")
		if 'module' in ckpt:
			ckpt = ckpt['module']

		msg = self.model.load_state_dict(ckpt, strict=False)
		self.model = self.model.eval()
		self.model = self.model.to(self.args.device).to(torch.bfloat16)

	def get_image_tensor(self, image):
		"""
		Get image tensor using model's vision processor.
		Tensor should have batch dimension.
		"""
		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image

		img = process_img(img=raw_image, device="cpu")
		return img

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
		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image

		img = process_img(img=raw_image, device=self.args.device)
		samples = {"image": img, "prompt": self.image_placeholder + " " + text}
		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			output_text = self.model.generate(
					samples,
					max_new_tokens=100,
					num_beams=5,
					length_penalty=0.0,
					repetition_penalty=1.0,
					temperature=0.9,
				)[0].strip()
		
		return output_text.split("\n")[0].strip()

	def generate_batch(self, 
					   images: torch.Tensor,
					   texts: List[str], 
					   batch_size, 
					   ):
		answers = []
		for i in tqdm(range(0, len(texts), batch_size)):
			img_batch = images[i : i + batch_size].to(self.args.device)
			text_batch = texts[i : i + batch_size]

			text_batch = [self.image_placeholder + " " + text for text in text_batch]
			samples = {"image": img_batch, "prompt": text_batch}
			with torch.cuda.amp.autocast(dtype=torch.bfloat16):
				output_texts = self.model.generate(
					samples,
					max_new_tokens=100,
					num_beams=5,
					length_penalty=0.0,
					repetition_penalty=1.0,
				)
			for out in output_texts:
				answers.append(out.split("\n")[0].strip())

		return answers
		
	def generate_image(self,
					   text,
					   image,
					   ):
		image, safety = self.image_generation_pipeline(
			[text],
			height=512,
			width=512,
			guidance_scale=7.5,
		)
		return image