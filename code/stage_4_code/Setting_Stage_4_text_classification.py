from code.base_class.setting import setting
from Dataset_TextClassification import Dataset_TextClassification

class Setting_Stage_4_text_classification(setting):
    def __init__(self, sName='Stage 4 IMDB Setting', sDescription=None):
        super().__init__(sName, sDescription)
        self.test_evaluators = None
        self.vocab = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestEvaluators=None, vocab=None):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
        self.test_evaluators = sTestEvaluators if sTestEvaluators else []
        self.vocab = vocab

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        train_data, test_data = loaded_data['train'], loaded_data['test']

        train_dataset = Dataset_TextClassification(train_data, self.vocab)
        test_dataset = Dataset_TextClassification(test_data, self.vocab)

        self.method.train(train_dataset)
        predictions, true_y = self.method.test(test_dataset)

        self.result.data = predictions
        self.result.save()

        self.evaluate.data = {
            'true_y': true_y,
            'pred_y': predictions
        }

        if self.test_evaluators:
            print("************ Testing Metrics ************")
            for evaluator in self.test_evaluators:
                evaluator.data = self.evaluate.data
                score = evaluator.evaluate()
                print(f"{evaluator.evaluate_name}: {score:.4f}")

        acc = self.evaluate.evaluate()
        print(f"\nTest Accuracy: {acc:.4f}")
        return acc, None
