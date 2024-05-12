import numpy as np
from tqdm import tqdm
from hemm.metrics.metric import HEMMMetric
from hemm.metrics.BARTScore.bart_score import BARTScorer

class BartScoreMetric(HEMMMetric):
    def __init__(self, device="cuda", batch_size=64):
        self.name = "Bart Score"
        self.device = device
        self.batch_size = batch_size
        self.scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    
    def bart_score_multi_ref(self, predictions, gts):
        """Bart Score for candidates with multiple references"""
        scores = []
        for i in tqdm(range(len(predictions))):
            pred = predictions[i]
            gt = gts[i]
            sc = self.scorer.multi_ref_score([pred], [gt], agg="max")
            scores.append(sc)

        scores = np.array(scores)
        return scores.mean()
    
    def bart_score_single_ref(self, predictions, gts):
        scores = self.scorer.score(predictions, gts, batch_size=self.batch_size)
        scores = np.array(scores)
        return scores.mean()

    def compute(self, 
                       predictions, 
                       ground_truth):        
        predictions, ground_truth = self.lower(predictions, ground_truth)

        # check for multiple references
        if isinstance(ground_truth[0], list):
            return self.bart_score_multi_ref(predictions, ground_truth)
        
        return self.bart_score_single_ref(predictions, ground_truth)
