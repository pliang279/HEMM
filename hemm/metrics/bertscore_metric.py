from hemm.metrics.metric import HEMMMetric
import evaluate
class BertScoreMetric(HEMMMetric):
    def __init__(self, device="cuda"):
        self.name = "Bert Score"
        self.device = device
    def compute(self, 
                       predictions, 
                       ground_truth):        
        bertscore = evaluate.load('bertscore', device=self.device)
        predictions, ground_truth = self.lower(predictions, ground_truth)

        results = bertscore.compute(predictions=predictions, references=ground_truth, 
                                    model_type="microsoft/deberta-large-mnli", lang="en", 
                                    rescale_with_baseline=True, device=self.device)
        
        results['precision'] = sum(results['precision'])/ len(results['precision'])
        results['recall'] = sum(results['recall'])/ len(results['recall'])
        results['f1'] = sum(results['f1'])/ len(results['f1'])
        return results
    