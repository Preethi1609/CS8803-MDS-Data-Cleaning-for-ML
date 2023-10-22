import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pre_process import preprocess_data_obesity, preprocess_data_runwalk
import sys


csv_path = sys.argv[1]
model = sys.argv[2]

df = pd.read_csv(csv_path)

if model=="runwalk":
    data = preprocess_data_runwalk(df)
elif model=="obesity":
    data = preprocess_data_obesity(df)

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Apply one-hot encoding
data = pd.get_dummies(data, columns=categorical_cols)

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# Convert to torch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

# Define a simple logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Compute accuracy
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    predictions = (predictions > 0.5).float()
    correct = (predictions == y_test_tensor).float().sum()
    accuracy = correct / len(y_test_tensor)
    print("accuracyy:", accuracy)
