from hemm.metrics.metric import HEMMMetric

from sklearn.metrics import accuracy_score

class AccuracyMetric(HEMMMetric):
    def compute_metric(self, 
                       ground_truth, 
                       predictions):
        return accuracy_score(predictions, ground_truth)