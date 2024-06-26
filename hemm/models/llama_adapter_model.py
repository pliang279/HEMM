import os
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from hemm.models.llama.llama_adapter import LLaMA_adapter
from hemm.utils.common_utils import shell_command

class LlamaAdapter(HEMMModel):
	def __init__(self,
				device="cuda",
				download_dir="./",
				**kwargs):
		super().__init__()
		self.device = device
		self.download_dir = f"{download_dir}/llama_adapter_weights"
		llama_dir = kwargs["llama_dir"]
		self.llama_ckpt_dir = os.path.join(llama_dir, "7B")
		self.llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
		self.model_path = f"{self.download_dir}/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth"
		self.prompt_format = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request using a single word or phrase.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
		
	def load_weights(self):
		if not os.path.exists(self.download_dir):
			shell_command(f"mkdir -p {self.download_dir}")
			shell_command(f"wget https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth -P {self.download_dir}")

		ckpt = torch.load(self.model_path, map_location="cpu")
		w_bias = True
		w_lora = True
		lora_rank = 16

		self.model = LLaMA_adapter(
				self.llama_ckpt_dir, self.llama_tokenzier_path,
				max_seq_len=1000, max_batch_size=32,
				clip_model='ViT-L/14',
				v_embed_dim=768, v_depth=8,
				v_num_heads=16, v_mlp_ratio=4.0,
				query_len=10, query_layer=31,
				w_bias=w_bias,
				w_lora=w_lora,
				lora_rank=lora_rank,
				w_new_gate=w_lora,)

		load_result = self.model.load_state_dict(ckpt['model'], strict=False)
		self.model = self.model.to(self.device)
		self.model.half()
		self.model.eval()
		self.preprocess = self.model.clip_transform

	def generate(self,
				text: Optional[str],
				image,
			) -> str:
		if not isinstance(image, Image.Image):
			raw_image = Image.open(image).convert("RGB")
		else:
			raw_image = image

		prompt = self.prompt_format.format_map({'instruction': text})
		image = self.preprocess(raw_image).unsqueeze(0).half().to(self.device)
		result = self.model.generate(image, [prompt], 
										max_gen_len=100, 
										temperature=0, 
										top_p=0.75)

		return result[0]
		
	def get_image_tensor(self, image):
		image = self.preprocess(image).unsqueeze(0).half()
		return image
	
	def answer_extractor(self, text, dataset_key):
		return text

	def generate_batch(self, 
					   images,
					   texts, 
					   batch_size, 
					   ):
		answers = []
		for i in tqdm(range(0, len(texts), batch_size)):
			img_batch = images[i : i + batch_size].to(self.device)
			text_batch = texts[i : i + batch_size]

			text_batch = [self.prompt_format.format_map({'instruction': text}) for text in text_batch]

			result = self.model.generate(img_batch, text_batch, 
										max_gen_len=100, 
										temperature=0, 
										top_p=0.75)
			
			answers += result
				
		return answers
