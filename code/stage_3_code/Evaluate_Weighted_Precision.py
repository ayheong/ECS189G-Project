from code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score

class Evaluate_Weighted_Precision(evaluate):
    def __init__(self):
        super().__init__()
        self.data = None
        self.evaluate_name = "Weighted Precision"

    def evaluate(self):
        # print('calculating precision...')
        return precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')