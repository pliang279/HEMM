from hemm.metrics.metric import HEMMMetric

import evaluate

class BleuMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        bleu = evaluate.load('bertscore')
        results = bleu.compute(predictions=predictions, references=ground_truth)
        results['precision'] = sum(results['precision'])/ len(results['precision'])
        results['recall'] = sum(results['recall'])/ len(results['recall'])
        results['f1'] = sum(results['f1'])/ len(results['f1'])
        return results