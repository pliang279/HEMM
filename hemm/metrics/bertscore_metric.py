from hemm.metrics.metric import HEMMMetric

import evaluate

class BertScoreMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        self.name = "Bert Score"
        bertscore = evaluate.load('bertscore')
        # results = bertscore.compute(predictions=predictions, references=ground_truth, lang='en')
        results = bertscore.compute(predictions=predictions, references=ground_truth, 
                                    model_type="microsoft/deberta-large-mnli", lang="en", 
                                    rescale_with_baseline=True)
        results['precision'] = sum(results['precision'])/ len(results['precision'])
        results['recall'] = sum(results['recall'])/ len(results['recall'])
        results['f1'] = sum(results['f1'])/ len(results['f1'])
        return results
    