from Dataset_Loader_ORL import Dataset_Loader_ORL
from Method_CNN_ORL import Method_CNN_ORL
from Evaluate_Accuracy import Evaluate_Accuracy
from Evaluate_Weighted_F1_Score import Evaluate_Weighted_F1_Score
from Evaluate_Weighted_Recall import Evaluate_Weighted_Recall
from Evaluate_Weighted_Precision import Evaluate_Weighted_Precision
from Setting_Stage_3_ORL import Setting_Stage_3_ORL
from Result_Saver import Result_Saver
import numpy as np
import torch

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    setting = Setting_Stage_3_ORL()

    result = Result_Saver()
    result.result_destination_folder_path = 'result/stage_3_result/'
    result.result_destination_file_name = 'cnn_orl_output'
    result.fold_count = 0

    # Prepare the experiment components
    setting.prepare(
        sDataset=Dataset_Loader_ORL(),
        sMethod=Method_CNN_ORL(),
        sResult=result,
        sEvaluate=Evaluate_Accuracy(),
        sTestEvaluators=[Evaluate_Weighted_Recall(), Evaluate_Weighted_Precision(), Evaluate_Weighted_F1_Score()]
    )

    # Run experiment
    setting.load_run_save_evaluate()
