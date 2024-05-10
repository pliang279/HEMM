import clip
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
from hemm.metrics.metric import HEMMMetric
import hemm.metrics.clipscore as clipscore

class CLIPMetric(HEMMMetric):
	def __init__(self, device):
		self.name = "Ref CLIP Score"
		self.device = device
		self.model, self.transform = clip.load("ViT-B/32", device)
		self.model.eval()

	def compute(self, input_imgs, preds, gts):
		preds, gts = self.lower(preds, gts)
		image_feats = clipscore.extract_all_images(input_imgs, self.model, self.device, 
											 batch_size=64, num_workers=8)

		# get image-text clipscore
		_, per_instance_image_text, candidate_feats = clipscore.get_clip_score(
			self.model, image_feats, preds, self.device)
	
		# get text-text clipscore
		_, per_instance_text_text = clipscore.get_refonlyclipscore(
				self.model, gts, candidate_feats, self.device)
		
		refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
	
		return refclipscores.mean()
