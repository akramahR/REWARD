import torch.nn as nn
import torch

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)          # Second fully connected layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImprovedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedNN, self).__init__()
        self.fc0 = nn.Linear(input_size, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(input_size, 256)  # Increased size
        self.bn1 = nn.BatchNorm1d(256)         # Added batch normalization
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)         # Added batch normalization
        self.fc3 = nn.Linear(128, num_classes) # Output layer
        self.dropout = nn.Dropout(0.5)         # Added dropout

    def forward(self, x):
        x = torch.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)                    # Dropout layer
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


'''
# Define a modified neural network model
class ImprovedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedNN, self).__init__()
        self.fc0 = nn.Linear(input_size, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(input_size, 256)  # Increased size
        self.bn1 = nn.BatchNorm1d(256)         # Added batch normalization
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)         # Added batch normalization
        self.fc3 = nn.Linear(128, num_classes) # Output layer
        self.dropout = nn.Dropout(0.2)         # Added dropout
        self.sml = nn.Softmax(dim=1)
    def forward(self, x):
        x = torch.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        #x = torch.relu(self.fc1(x))
        x = self.dropout(x)                    # Dropout layer
        x = torch.relu(self.bn2(self.fc2(x)))
        #x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sml(x)
        return x

'''