from hemm.metrics.metric import HEMMMetric

import evaluate

class RougeMetric(HEMMMetric):
	def __init__(self):
		self.name = "Rouge"
	def compute(self, predictions, ground_truth):
		rouge = evaluate.load('rouge')
		predictions, ground_truth = self.lower(predictions, ground_truth)
		ground_truth = [[x] for x in ground_truth]
		results = rouge.compute(predictions=predictions, references=ground_truth)
		return results
		
		