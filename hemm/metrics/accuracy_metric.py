from hemm.metrics.metric import HEMMMetric

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AccuracyMetric(HEMMMetric):
    def __init__(self):
        self.name = "Accuracy"
    def compute(self, 
                       ground_truth, 
                       predictions):        
        return accuracy_score(predictions, ground_truth)

class PrecisionMetric(HEMMMetric):
    def __init__(self):
        self.name = "Precision"
    def compute(self, 
                       ground_truth, 
                       predictions):
        return precision_score(predictions, ground_truth, average="macro")

class RecallMetric(HEMMMetric):
    def __init__(self):
        self.name = "Recall"
    def compute(self, 
                       ground_truth, 
                       predictions):
        return recall_score(predictions, ground_truth, average="macro")

class F1ScoreMetric(HEMMMetric):
    def __init__(self):
        self.name = "F1-Score"
    def compute(self, 
                    ground_truth, 
                       predictions):
        return f1_score(predictions, ground_truth, average="macro")
