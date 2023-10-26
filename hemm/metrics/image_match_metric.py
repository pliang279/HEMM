import clip
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
from hemm.metrics.metric import HEMMMetric
from scipy import spatial

class MSEMetric(HEMMMetric):
	def compute(self, 
				img_preds, 
				img_gts):
		self.name = "Image MSE"
		crit = nn.MSELoss()
		error = 0
		for pred, gt in zip(img_preds, img_gts):
			gt_img = Image.open(gt).convert("RGB")
			pred_img = pred.resize(gt_img.size)
			gt_img = transforms.ToTensor()(gt_img)
			pred_img = transforms.ToTensor()(pred_img)
			error += crit(pred_img, gt_img).detach().cpu().numpy().item()

		return error / len(img_preds)
	
class CLIPIMetric(HEMMMetric):
	def __init__(self, device):
		self.name = "CLIP-I Score"
		self.device = device
		self.model, self.transform = clip.load("ViT-B/32", device)

	def encode(self, image):
		image_input = self.transform(image).unsqueeze(0).to(self.device)
		with torch.no_grad():
			image_features = self.model.encode_image(image_input).detach().cpu().float()

		return image_features

	def compute(self,
				img_preds,
				img_gts):        
		"""
		Calculate CLIP-I score, the cosine similarity between the generated image and the ground truth image
		"""
		eval_score = 0
		for pred, gt in zip(img_preds, img_gts):
			gt_img = Image.open(gt).convert("RGB")
			gt_features = self.encode(gt_img)
			pred_features = self.encode(pred)
			
			similarity = 1 - spatial.distance.cosine(pred_features.view(pred_features.shape[1]),
													gt_features.view(gt_features.shape[1]))
			if similarity > 1 or similarity < -1:
				raise ValueError(" strange similarity value")
			eval_score = eval_score + similarity
			
		return eval_score / len(img_preds)

