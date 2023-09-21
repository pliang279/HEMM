from hemm.metrics.metric import HEMMMetric

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AccuracyMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        self.name = "Accuracy"
        return accuracy_score(predictions, ground_truth)

class PrecisionMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        self.name = "Precision"
        return precision_score(predictions, ground_truth, average="macro")

class RecallMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        self.name = "Recall"
        return recall_score(predictions, ground_truth, average="macro")


class F1ScoreMetric(HEMMMetric):
    def compute(self, 
                       ground_truth, 
                       predictions):
        self.name = "F1-Score"
        return f1_score(predictions, ground_truth, average="macro")
