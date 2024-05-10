import os
import torch
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm
from hemm.models.model import HEMMModel
from hemm.models.llama.llama_adapter import LLaMA_adapter

class LlamaAdapter(HEMMModel):
	def __init__(self,
			  	llama_dir="/work/agoindan/.cache/llama_weights",
				llama_type="7B",
				model_path = "/home/agoindan/LLaMA-Adapter/llama_adapter_v2_multimodal7b/ckpts/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth"
				 ):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.llama_ckpt_dir = os.path.join(llama_dir, llama_type)
		self.llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
		self.model_path = model_path
		
	def load_weights(self):
		ckpt = torch.load(self.model_path, map_location=self.device)
		w_bias = True
		w_lora = True
		lora_rank = 16

		self.model = LLaMA_adapter(
				self.llama_ckpt_dir, self.llama_tokenzier_path,
				max_seq_len=512, max_batch_size=32,
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

		image = self.preprocess(raw_image).unsqueeze(0).half().to(self.device)
		result = self.model.generate(image, [text], 
										max_gen_len=100, 
										temperature=0, 
										top_p=0.75)

		return result[0]
		
	def get_image_tensor(self, image):
		image = self.preprocess(image).unsqueeze(0).half().to(self.device)
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
			img_batch = images[i : i + batch_size]
			text_batch = texts[i : i + batch_size]

			result = self.model.generate(img_batch, text_batch, 
										max_gen_len=100, 
										temperature=0, 
										top_p=0.75)
			
			answers += result
				
		return answers
