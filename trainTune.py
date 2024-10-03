import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models import ImprovedNN

# Load the extracted training and validation data
train_data = np.load("trainData.npy")
val_data = np.load("valData.npy")

# Split the data into features and labels
X_train, y_train = train_data[:, :512], train_data[:, 512:]  # First 512 columns are features
X_val, y_val = val_data[:, :512], val_data[:, 512:]          # Same for validation data

# Standardize the features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Keep as float for one-hot encoding
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)      # Keep as float for one-hot encoding

# Convert one-hot encoded labels to class indices for training and evaluation
y_train_indices = torch.argmax(y_train_tensor, dim=1)
y_val_indices = torch.argmax(y_val_tensor, dim=1)


# Define the neural network model, loss function, and optimizer
input_size = X_train.shape[1]
num_classes = y_train_tensor.shape[1]  # Number of classes is the number of columns in one-hot encoded labels
#model = SimpleNN(input_size, num_classes)
model = ImprovedNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train_tensor, y_train_indices = X_train_tensor.to(device), y_train_indices.to(device)
X_val_tensor, y_val_indices = X_val_tensor.to(device), y_val_indices.to(device)

# Train the model
num_epochs = 800
batch_size = 24

best_accuracy = 0.0  # Variable to store the best validation accuracy

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])

    epoch_loss = 0.0
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_indices[indices]  # Use indices for class labels

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
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, val_predictions = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_val_indices.cpu(), val_predictions.cpu())

    #print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Check if the current validation accuracy is the best
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(f"New best accuracy: {best_accuracy * 100:.2f}% - Saving model")
        torch.save(model.state_dict(), "best_nn_model.pth")  # Save the model with the best accuracy
