import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from rnn_net_text_generation import rnn_net_text_generation
import matplotlib.pyplot as plt


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
        epoch_losses = []
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        # Plot loss curve
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs (Text Generation)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('result/stage_4_result/training_loss_plot_generation.png')
        plt.close()

    def generate(self, prompt_words, vocab, max_len=75):
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
                temperature = 0.5
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                # next_token = torch.argmax(next_token_logits).item()

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
