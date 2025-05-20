import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from rnn_net_text_generation import rnn_net_text_generation


def collate_fn_gen(batch):
    sequences = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences


class Method_RNN_Text_Generation:
    def __init__(self, vocab_size, embedding_matrix, embedding_dim=100, hidden_dim=128):
        self.model = rnn_net_text_generation(vocab_size, embedding_dim, hidden_dim, embedding_matrix)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.003
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn_gen)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")

    def generate(self, prompt_words, vocab, max_len=50):
        self.model.eval()
        idx_to_word = {v: k for k, v in vocab.items()}
        input_indices = [vocab.get(word, vocab['<unk>']) for word in prompt_words]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        generated = input_indices[:]

        hidden = None
        with torch.no_grad():
            for _ in range(max_len):
                outputs, hidden = self.model(input_tensor, hidden)
                next_token_logits = outputs[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()

                next_word = idx_to_word.get(next_token, '<unk>')
                if next_word == '<pad>':
                    continue

                generated.append(next_token)

                if next_word in ['.', '?', '!']:
                    break

                input_tensor = torch.tensor([[next_token]], dtype=torch.long).to(self.device)

        generated_words = [idx_to_word.get(idx, '<unk>') for idx in generated]
        result = ' '.join(generated_words)
        print("Generated:", result)
        return result
