import torch
import torch.nn as nn

class rnn_net_text_classification(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, output_dim=2, embedding_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
            self.embedding.weight.requires_grad = False

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden_cat)
        return self.fc(out)



