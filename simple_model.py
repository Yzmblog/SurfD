import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=1, output_size=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return  x
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.sigmoid(out)
        # return out