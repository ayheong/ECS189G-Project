import os
import torch
from code.base_class.dataset import dataset
from collections import Counter
import csv

class Dataset_TextGeneration(dataset):
    def __init__(self, data=None, vocab=None, dName='Movie Dataset', dDescription=''):
        super().__init__(dName, dDescription)
        self.vocab = vocab
        self.raw_data = data or []
        self.dataset_source_folder_path = None

    @staticmethod
    def build_vocab(csv_path, min_freq=1):
        counter = Counter()
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                joke = row['Joke'].strip()
                if joke:
                    words = joke.split()
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

        jokes_path = os.path.join(base_path, "data")
        data = []
        with open(jokes_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                joke = row['Joke'].strip()
                if joke:
                    data.append({'text': joke, 'label': 0})

        self.raw_data = data
        self.data = {'train': data}
        return self.data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        text_tokens = item['text'].split()
        encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in text_tokens]

        input_seq = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)
        return input_seq, target_seq

