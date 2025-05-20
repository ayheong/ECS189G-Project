import pickle
import torch
from Dataset_TextGeneration import Dataset_TextGeneration
from Method_RNN_Text_Generation import Method_RNN_Text_Generation
from Setting_Stage_4_text_generation import Setting_Stage_4_text_generation

if __name__ == '__main__':
    try:
        with open('cached_vocab_gen.pkl', 'rb') as f:
            vocab = pickle.load(f)
        with open('cached_glove_gen.pkl', 'rb') as f:
            glove = pickle.load(f)
        embedding_matrix = torch.load('cached_embedding_gen.pt')
        print("Loaded cached vocab and embeddings for text generation.")
    except FileNotFoundError:
        print("Cache not found. Building vocab and embeddings...")
        vocab = Dataset_TextGeneration.build_vocab('data/stage_4_data/text_generation/data')
        glove = Dataset_TextGeneration.load_glove_embeddings('data/stage_4_data/glove.6B.100d.txt')
        embedding_matrix = Dataset_TextGeneration.build_embedding_matrix(vocab, glove)

        with open('cached_vocab_gen.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        with open('cached_glove_gen.pkl', 'wb') as f:
            pickle.dump(glove, f)
        torch.save(embedding_matrix, 'cached_embedding_gen.pt')
        print("Cache saved for future runs.")

    vocab_size = len(vocab)

    setting = Setting_Stage_4_text_generation()
    setting.prepare(
        sDataset=Dataset_TextGeneration(),
        sMethod=Method_RNN_Text_Generation(vocab_size, embedding_matrix),
        sResult=None,
        sEvaluate=None,
        vocab=vocab,
        prompt_words=["take", "we", "under", "all", "boat"]
    )
    setting.dataset.dataset_source_folder_path = 'data/stage_4_data/text_generation'
    setting.load_run_save_evaluate()
