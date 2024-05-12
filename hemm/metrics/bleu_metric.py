from hemm.metrics.metric import HEMMMetric

import evaluate

class BleuMetric(HEMMMetric):
    def __init__(self):
        self.name = "Blue Score"
    def compute(self, 
                       predictions, 
                       ground_truth):
        bleu = evaluate.load('bleu')
        ground_truth = [[x] for x in ground_truth]
        results = bleu.compute(predictions=predictions, references=ground_truth)
        return results
    