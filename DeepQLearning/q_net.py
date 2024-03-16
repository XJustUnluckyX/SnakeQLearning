import torch
import torch.nn as nn
import os

# E' stato utilizzato come riferimento il seguente progetto
# https://github.com/patrickloeber/snake-ai-pytorch/tree/main , poi rifattorizzato

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # Rete feed-forward
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save_model(self, file_name='mdl.pth'):
        model_path = './mdl'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)
