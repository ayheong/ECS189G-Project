from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Evaluate_Weighted_F1_Score import Evaluate_Weighted_F1_Score
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Stage_2 import Setting_Stage_2
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_2_code.Evaluate_Weighted_Recall import Evaluate_Weighted_Recall
from code.stage_2_code.Evaluate_Weighted_Precision import Evaluate_Weighted_Precision
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage 2', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_train_source_file_name = 'train.csv'
    data_obj.dataset_test_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Stage_2('Stage 2 settings', '')

    evaluate_obj = Evaluate_Accuracy('MLP accuracy', '')

    accuracy_eval = Evaluate_Accuracy('Accuracy', '')
    f1_eval = Evaluate_Weighted_F1_Score('Weighted F1 Score', '')
    precision_eval = Evaluate_Weighted_Precision('Precision', '')
    recall_eval = Evaluate_Weighted_Recall('Recall', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(
        data_obj,
        method_obj,
        result_obj,
        evaluate_obj,
        [accuracy_eval, f1_eval, precision_eval, recall_eval]
    )
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    