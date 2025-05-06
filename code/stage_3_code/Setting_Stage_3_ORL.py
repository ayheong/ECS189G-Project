from code.base_class.setting import setting
from Dataset_ORL import Dataset_ORL

class Setting_Stage_3_ORL(setting):
    def __init__(self, sName='Stage 3 ORL Setting', sDescription=None):
        super().__init__(sName, sDescription)
        self.test_evaluators = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestEvaluators=None):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
        self.test_evaluators = sTestEvaluators if sTestEvaluators else []

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        train_data, test_data = loaded_data['train'], loaded_data['test']

        train_dataset = Dataset_ORL(train_data)
        test_dataset = Dataset_ORL(test_data)

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
