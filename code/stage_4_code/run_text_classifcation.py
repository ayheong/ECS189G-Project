from Dataset_TextClassification import Dataset_TextClassification
from Method_RNN_Text_Classification import Method_RNN_Text_Classification
from Evaluate_Accuracy import Evaluate_Accuracy
from Evaluate_Weighted_F1_Score import Evaluate_Weighted_F1_Score
from Evaluate_Weighted_Recall import Evaluate_Weighted_Recall
from Evaluate_Weighted_Precision import Evaluate_Weighted_Precision
from Setting_Stage_4_text_classification import Setting_Stage_4_text_classification
from Result_Saver import Result_Saver
import numpy as np
import torch
import pickle


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    try:
        with open('cached_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        with open('cached_glove.pkl', 'rb') as f:
            glove = pickle.load(f)
        embedding_matrix = torch.load('cached_embedding.pt')
        print("Loaded cached vocab and embeddings")
    except FileNotFoundError:
        print("Cache not found, rebuilding...")
        vocab = Dataset_TextClassification.build_vocab('data/stage_4_data/text_classification')
        glove = Dataset_TextClassification.load_glove_embeddings('data/stage_4_data/glove.6B.100d.txt')
        embedding_matrix = Dataset_TextClassification.build_embedding_matrix(vocab, glove)
        with open('cached_vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        with open('cached_glove.pkl', 'wb') as f:
            pickle.dump(glove, f)
        torch.save(embedding_matrix, 'cached_embedding.pt')
        print("Cache saved.")

    vocab_size = len(vocab)  # 108557
    # print('vocab size:', vocab_size)

    setting = Setting_Stage_4_text_classification()

    result = Result_Saver()
    result.result_destination_folder_path = 'result/stage_4_result/'
    result.result_destination_file_name = 'rnn_text_classification_output'

    # Prepare the experiment components
    setting.prepare(
        sDataset=Dataset_TextClassification(),
        sMethod=Method_RNN_Text_Classification(vocab_size=vocab_size, embedding_matrix=embedding_matrix),
        sResult=result,
        sEvaluate=Evaluate_Accuracy(),
        sTestEvaluators=[Evaluate_Weighted_Recall(), Evaluate_Weighted_Precision(), Evaluate_Weighted_F1_Score()],
        vocab=vocab
    )

    setting.dataset.dataset_source_folder_path = 'data/stage_4_data/text_classification'

    # Run experiment
    setting.load_run_save_evaluate()


