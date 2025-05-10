from Dataset_Loader_MNIST import Dataset_Loader_MNIST
from Method_CNN_MNIST import Method_CNN_MNIST
from Evaluate_Accuracy import Evaluate_Accuracy
from Evaluate_Weighted_F1_Score import Evaluate_Weighted_F1_Score
from Evaluate_Weighted_Recall import Evaluate_Weighted_Recall
from Evaluate_Weighted_Precision import Evaluate_Weighted_Precision
from Setting_Stage_3_MNIST import Setting_Stage_3_MNIST
from Result_Saver import Result_Saver
import numpy as np
import torch

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    setting = Setting_Stage_3_MNIST()

    result = Result_Saver()
    result.result_destination_folder_path = 'result/stage_3_result/'
    result.result_destination_file_name = 'cnn_mnist_output'
    result.fold_count = 0

    setting.prepare(
        sDataset=Dataset_Loader_MNIST(),
        sMethod=Method_CNN_MNIST(),
        sResult=result,
        sEvaluate=Evaluate_Accuracy(),
        sTestEvaluators=[Evaluate_Weighted_Recall(), Evaluate_Weighted_Precision(), Evaluate_Weighted_F1_Score()]
    )

    setting.load_run_save_evaluate()
