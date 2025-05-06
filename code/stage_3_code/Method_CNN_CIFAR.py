from code.base_class.method import method
from cnn_net_CIFAR import Net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

class Method_CNN_CIFAR(method):
    def __init__(self):
        super().__init__('CNN_CIFAR', '')
        self.model = Net()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 10
        self.batch_size = 64
        self.learning_rate = 0.001
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataset):
        self.model.to(self.device)
        self.model.train()
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

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
                self.optimizer.step()
            avg_loss = running_loss / len(trainloader)
            epoch_losses.append(avg_loss)
            print(f'Epoch {epoch+1} Loss: {avg_loss:.4f}')

        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        save_dir = 'result/stage_3_result'
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'training_loss_plot_CIFAR.png')
        plt.savefig(plot_path)
        plt.close()

    def test(self, test_dataset):
        self.model.eval()
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
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
