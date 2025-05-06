from Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
from Method_CNN_CIFAR import Method_CNN_CIFAR
from Evaluate_Accuracy import Evaluate_Accuracy
from Setting_Stage_3_CIFAR import Setting_Stage_3_CIFAR
from Result_Saver import Result_Saver
import numpy as np
import torch

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    setting = Setting_Stage_3_CIFAR()

    result = Result_Saver()
    result.result_destination_folder_path = 'result/stage_3_result/'
    result.result_destination_file_name = 'cnn_cifar_output'
    result.fold_count = 0

    # Prepare the experiment components
    setting.prepare(
        sDataset=Dataset_Loader_CIFAR(),
        sMethod=Method_CNN_CIFAR(),
        sResult=result,
        sEvaluate=Evaluate_Accuracy()
    )

    # Run experiment
    setting.load_run_save_evaluate()
