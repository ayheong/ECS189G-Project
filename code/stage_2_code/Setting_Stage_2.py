from code.base_class.setting import setting
import numpy as np

class Setting_Stage_2(setting):

    def __init__(self, sName='Stage 2 Setting', sDescription=None):
        super().__init__(sName, sDescription)
        self.test_evaluators = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestEvaluators=None):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
        self.test_evaluators = sTestEvaluators if sTestEvaluators else []

    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        X_train, y_train, X_test, y_test = loaded_data['X_train'], loaded_data['y_train'], loaded_data['X_test'], loaded_data['y_test']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        if self.test_evaluators:
            print("************ Testing Metrics ************")
            for evaluator in self.test_evaluators:
                evaluator.data = learned_result
                score = evaluator.evaluate()
                print(f"{evaluator.evaluate_name}: {score:.4f}")

        return self.evaluate.evaluate(), None

        