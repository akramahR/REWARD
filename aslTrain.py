import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.src.metrics.accuracy_metrics import accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
from models import ImprovedNN
from collections import Counter

# Load the extracted training and validation data
train_data = np.load("trainData.npy")
val_data = np.load("valData.npy")

# Split the data into features and labels
X_train, y_train = train_data[:, :512], train_data[:, 512:]  # First 512 columns are features
X_val, y_val = val_data[:, :512], val_data[:, 512:]          # Same for validation data

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Keep as float for one-hot encoding
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)      # Keep as float for one-hot encoding

y_train_indices = torch.argmax(y_train_tensor, dim=1)
y_val_indices = torch.argmax(y_val_tensor, dim=1)

# Define the neural network model, loss function, and optimizer
input_size = X_train.shape[1]
num_classes = y_train_tensor.shape[1]  # Number of classes is the number of columns in one-hot encoded labels
# model = SimpleNN(input_size, num_classes)
model = ImprovedNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

# Move the model to GPU if available
model.to(device="cuda:0")
X_train_tensor, y_train_indices = X_train_tensor.to(device="cuda:0"), y_train_indices.to(device="cuda:0")
X_val_tensor, y_val_indices = X_val_tensor.to(device="cuda:0"), y_val_indices.to(device="cuda:0")

# Train the model
num_epochs = 2000
batch_size = 256
best = 0

for epoch in range(num_epochs):
    model.train()
    #permutation = torch.randperm(X_train_tensor.size()[0])

    epoch_loss = 0.0
    #for i in range(0, X_train_tensor.size()[0]):
    #indices = permutation[i:i+batch_size]
    batch_X, batch_y = X_train_tensor, y_train_indices  # Use indices for class labels

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()

    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluate the model on the validation set
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_predictions = torch.argmax(val_outputs, dim=1)
        y_val_max = torch.argmax(y_val_indices, dim=1)

        correct =0
        for j, prediction in enumerate(val_predictions):
            if prediction == y_val_max[j]:
                correct = correct+1

        accuracy = correct/len(val_predictions)
        if accuracy > best:
            best =accuracy
            print("accuracy:" + str(best))

# Save the trained model
torch.save(model.state_dict(), "simple_nn_model.pth")


