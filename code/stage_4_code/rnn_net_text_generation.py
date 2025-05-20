import torch.nn as nn
import torch

class rnn_net_text_generation(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=512, embedding_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
            self.embedding.weight.requires_grad = True
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.dropout = nn.Dropout(0.3)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)
        output = self.layernorm(output)
        logits = self.fc(output)
        return logits, hidden