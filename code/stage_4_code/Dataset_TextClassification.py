import os
import torch
from code.base_class.dataset import dataset
from collections import Counter

class Dataset_TextClassification(dataset):
    def __init__(self, data=None, vocab=None, dName='Movie Dataset', dDescription=''):
        super().__init__(dName, dDescription)
        self.vocab = vocab
        self.raw_data = data or []
        self.dataset_source_folder_path = None

    @staticmethod
    def build_vocab(base_dir, min_freq=1):
        counter = Counter()
        for split in ['train']:
            for label in ['pos', 'neg']:
                folder = os.path.join(base_dir, split, label)
                for filename in os.listdir(folder):
                    if filename.endswith('.txt'):
                        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                            words = f.read().split()
                            counter.update(words)

        specials = ['<pad>', '<unk>']
        vocab = {token: idx for idx, token in enumerate(specials)}
        idx = len(vocab)

        for word, freq in counter.items():
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1

        return vocab

    @staticmethod
    def load_glove_embeddings(glove_path):
        embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
                embeddings[word] = vector
        return embeddings

    @staticmethod
    def build_embedding_matrix(vocab, glove_embeddings, embedding_dim=100):
        matrix = torch.randn(len(vocab), embedding_dim) * 0.05
        matrix[vocab['<pad>']] = torch.zeros(embedding_dim)

        for word, idx in vocab.items():
            if word in glove_embeddings:
                matrix[idx] = glove_embeddings[word]
        return matrix

    def load(self):
        base_path = self.dataset_source_folder_path
        if base_path is None:
            raise ValueError("dataset_source_folder_path must be set before calling load()")

        dataset_split = {}
        for split in ['train', 'test']:
            data = []
            for label_name, label_value in [('pos', 1), ('neg', 0)]:
                folder_path = os.path.join(base_path, split, label_name)
                for filename in os.listdir(folder_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(folder_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            data.append({'text': text, 'label': label_value})
            dataset_split[split] = data

        self.data = dataset_split
        return dataset_split

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        text_tokens = item['text'].split()
        encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in text_tokens]
        label = item['label']
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
