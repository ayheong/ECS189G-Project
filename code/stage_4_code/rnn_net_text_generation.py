import torch.nn as nn
import torch

class rnn_net_text_generation(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, embedding_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
            self.embedding.weight.requires_grad = False  
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output)
        return logits, hidden