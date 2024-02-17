import sys
sys.path.append("/home/agoindan/HEMM")

import os
os.chdir("/work/agoindan/.cache")
os.environ['TRANSFORMERS_CACHE'] = "/work/agoindan/.cache"
os.environ['TORCH_HUB'] = "/work/agoindan/.cache"
os.environ['XDG_CACHE_HOME'] = "/work/agoindan/.cache"

import torch
from PIL import Image
import pickle 
from tqdm import tqdm 
from transformers import FuyuProcessor, FuyuForCausalLM

from hemm.utils.base_utils import load_dataset_evaluator

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")

dataset_evaluator = load_dataset_evaluator(kaggle_api_path="/home/agoindan/")
no_batch = ["nlvr", "nlvr2", "winogroundVQA", "magic_brush"]
no_eval = ["irfl", "decimer", "enrico", "gqa", "nlvr", "nlvr2"]

for dset in dataset_evaluator:
	if os.path.exists(f"fuyu_{dset}.pkl"):
		continue
	
	print(f"Evaluating for {dset}")

	predictions = []
	for img, text, gt in tqdm(zip(images, texts, gts), total=len(images)):
		img = img.convert("RGB")
		inputs = processor(text=text, images=img, return_tensors="pt").to("cuda:0")

		generation_output = model.generate(**inputs, max_new_tokens=100)
		generation_text = processor.batch_decode(generation_output[:, -100:], skip_special_tokens=True)
		predictions.append(generation_text[0].split("\x04")[-1].strip())

		final = {
			"predictions": predictions,
			"gts": gts
		}
		pickle.dump(final, open(f"fuyu_{dset}.pkl", "wb"))
