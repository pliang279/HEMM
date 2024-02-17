import sys
sys.path.append("/home/agoindan/HEMM")

import os
os.chdir("/work/agoindan/.cache")
os.environ['TRANSFORMERS_CACHE'] = "/work/agoindan/.cache"
os.environ['TORCH_HUB'] = "/work/agoindan/.cache"
os.environ['XDG_CACHE_HOME'] = "/work/agoindan/.cache"

import argparse
import json

import torch
import pickle 
from models.modeling_emu import Emu
from utils import process_img, process_video
from hemm.utils.base_utils import load_dataset_evaluator
from tqdm import tqdm

def parse_args():
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
		default='/work/agoindan/.cache/emu/Emu-pretrain.pt',
		help="Emu ckpt path",
	)
	args = parser.parse_args('')

	return args

args = parse_args()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
with open(f'/home/agoindan/Emu/models/Emu-14B.json', "r", encoding="utf8") as f:
	model_cfg = json.load(f)

model = Emu(**model_cfg, cast_dtype=torch.float, args=args)
ckpt = torch.load(args.ckpt_path, map_location=args.device)
if 'module' in ckpt:
	ckpt = ckpt['module']

msg = model.load_state_dict(ckpt, strict=False)
model.eval()
model = model.to(args.device).to(torch.bfloat16)

system = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."
dataset_evaluator = load_dataset_evaluator(kaggle_api_path="/home/agoindan/")

for dset in dataset_evaluator:
	if not os.path.exists(f"{dset}.pkl"):
		continue

	if os.path.exists(f"emu_{dset}.pkl"):
		continue
	  
	print(f"Evaluating for {dset}")
	data = pickle.load(open(f"{dset}.pkl", "rb"))
	images = data["images"]
	texts = data["texts"]
	gts = data["gts"]

	predictions = []
	for img, text, gt in tqdm(zip(images, texts, gts), total=len(images)):
		img = process_img(img=img, device=args.device)
		text_seq = "[IMG]" + "<image>" * 32 + "[/IMG]" + text
		prompt = f"{system} [USER]: {text_seq} [ASSISTANT]:".strip()
		samples = {"image": torch.cat([img], dim=0), "prompt": prompt}
		output_text = model.generate(
				samples,
				max_new_tokens=500,
				num_beams=5,
				length_penalty=0.0,
				repetition_penalty=1.0,
			)[0].strip()
		
		predictions.append(output_text)
	
	try:
		results = {}
		metrics = dataset_evaluator[dset].metrics
		for metric in metrics:
			results[metric.name] = metric.compute(gts, predictions)
		
		final = {
			"predictions": predictions,
			"metrics": results, 
			"gts": gts
		}
	except:
		final = {
			"predictions": predictions,
			"gts": gts
		}
	
	pickle.dump(final, open(f"emu_{dset}.pkl", "wb"))
