import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from dataloader import X, Y

X = X.astype(np.float32)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

print(X.shape, Y.shape)

# preparing data for a multi-label classification
mlb = MultiLabelBinarizer()
Y_encoded = mlb.fit_transform(Y)
Y_encoded = torch.tensor(Y_encoded).float()
# Y_encoded = Y_encoded.astype(np.float32)

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        # (input channels, output channels, kernel size, padding)
        self.conv1 = nn.Conv1d(12, 64, 3, 1)
        self.conv2 = nn.Conv1d(64, 128, 3, 1)
        # (kernel, stride)
        self.pool = nn.MaxPool1d(2, 2)
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True)
        self.lstm_output_size = 128 * 2 * (1000//2)
        self.fc1 = nn.Linear(self.lstm_output_size, 512) # we cut this to 500 bc the max pool layer will half the size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(mlb.classes_))
        self.drop = nn.Dropout(0.5)
    
    def forward(self, X):
        X = torch.Tensor(X)
        X = X .transpose(1, 2) # have to reshape this for the Conv1d
        print(1, X.shape)
        X = torch.relu(self.conv1(X))
        print(2, X.shape)
        X = torch.relu(self.conv2(X))
        print(3, X.shape)
        X = self.pool(X)
        print(4, X.shape)
        X = X.transpose(1, 2) # switching back to the original size for the LSTM
        print(5, X.shape)
        lstm_out, _ = self.lstm(X)
        print(6, X.shape)

        # dimension error here
        lstm_out = lstm_out.contiguous().view(X.size(0), -1)
        print(7, X.shape)
        X = self.fc1(lstm_out)
        print(8, X.shape)
        X = torch.relu(X)
        print(9, X.shape)
        X = self.drop(X)
        print(10, X.shape)
        X = self.fc2(X)
        print(11,X.shape)
        X = torch.relu(X)
        print(12, X.shape)
        X = self.drop(X)
        print(13, X.shape)
        out = self.fc3(X)
        return out
    
model = CRNN()

crit = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X), batch_size):
        inputs = X[i:i+batch_size]
        targets = Y_encoded[i:i+batch_size]

        optimizer.zero_grad()
        output = model(inputs)
        loss = crit(output, targets)
        loss.backward()
        optimizer.step()
    
    print(f'epoch [{epoch+1}/{num_epochs}], loss: {loss.item():.4f}')


