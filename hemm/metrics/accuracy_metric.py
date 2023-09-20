from hemm.metrics.metric import HEMMMetric

from sklearn.metrics import accuracy_score

class AccuracyMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        self.name = "Accuracy"
        return accuracy_score(predictions, ground_truth)
    
