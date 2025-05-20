from code.base_class.evaluate import evaluate
from sklearn.metrics import recall_score

class Evaluate_Weighted_Recall(evaluate):
    def __init__(self):
        super().__init__()
        self.data = None
        self.evaluate_name = "Weighted Recall"

    def evaluate(self):
        # print('calculating recall...')
        return recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')