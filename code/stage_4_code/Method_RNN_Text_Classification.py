from code.base_class.method import method
from rnn_net_text_classifcation import rnn_net_text_classification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.tensor(labels, dtype=torch.long)

class Method_RNN_Text_Classification(method):
    def __init__(self, vocab_size, embedding_matrix=None, embedding_dim=100, hidden_dim=128, output_dim=2):
        super().__init__('RNN_Text', '')
        self.model = rnn_net_text_classification(vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights=embedding_matrix)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.001
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataset):
        self.model.to(self.device)
        self.model.train()
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        epoch_losses = []
        for epoch in range(self.epochs):
            running_loss = 0.0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

            avg_loss = running_loss / len(trainloader)
            epoch_losses.append(avg_loss)
            print(f'Epoch {epoch + 1} | Loss: {avg_loss:.4f}')

            # Plot loss curve
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        save_dir = 'result/stage_4_result'
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'training_loss_plot_text.png')
        plt.savefig(plot_path)
        plt.close()

    def test(self, test_dataset):
        self.model.eval()
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        predictions = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        return predictions, true_labels
