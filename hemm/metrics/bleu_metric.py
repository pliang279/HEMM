from hemm.metrics.metric import HEMMMetric

import evaluate

class BleuMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        bleu = evaluate.load('bleu')
        ground_truth = [[x] for x in ground_truth]
        results = bleu.compute(predictions=predictions, references=ground_truth)
        return results