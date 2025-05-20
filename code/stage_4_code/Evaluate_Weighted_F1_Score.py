from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score

class Evaluate_Weighted_F1_Score(evaluate):
    def __init__(self):
        super().__init__()
        self.data = None
        self.evaluate_name = "Weighted F1 Score"

    def evaluate(self):
        # print('calculating f1 score...')
        return f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')