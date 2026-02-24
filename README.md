# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
To design and implement a Recurrent Neural Network (RNN) model that learns patterns from historical stock price data and predicts future stock closing prices. The goal is to evaluate how well sequential deep learning models can capture time-dependent financial trends and compare predicted values with actual stock prices.


The dataset consists of historical stock prices of Google divided into two files:

Training Dataset — trainset.csv

Used to train the RNN model.
Contains historical daily stock information such as:

Date

Open price

High price

Low price

Close price

Volume

Only the Close price column is used for prediction because it represents the final market value of the stock for that day.

## Design Steps

### Step 1:
Load Dataset
Import required libraries and read the training and testing CSV files containing historical stock prices.

### Step 2:

Preprocess Data
Extract the Close price column, normalize values using MinMaxScaler, and convert data into sequences of past 60 days to predict the next day.
### Step 3:
Prepare Training Data
Convert sequences into tensors and create DataLoader batches for efficient training.
### Step 4:
Build and Train Model
Define an LSTM-based RNN architecture, set loss function (MSE) and optimizer (Adam), and train the model over multiple epochs.
### Step 5:
Predict and Evaluate
Use the trained model to predict test data prices, convert predictions back to original scale, and compare them with actual values using a graph.


## Program
#### Name: Somalaraju Rohini
#### Register Number: 212224240156
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take output from last timestep
        out = out[:, -1, :]

        # Final output layer
        out = self.fc(out)
        return out





model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the Model

epochs = 20
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")






```

## Output

### True Stock Price, Predicted Stock Price vs time
<img width="745" height="112" alt="Screenshot 2026-02-24 161906" src="https://github.com/user-attachments/assets/6b06460b-d076-4e0e-921d-ec253217c610" />





<img width="945" height="505" alt="Screenshot 2026-02-24 161917" src="https://github.com/user-attachments/assets/d339c3c9-dbc2-482e-876d-b85e3641c0c3" />





<img width="501" height="436" alt="Screenshot 2026-02-24 161925" src="https://github.com/user-attachments/assets/4e16de08-feb3-4295-b630-5e3ee025114a" />




<img width="893" height="633" alt="Screenshot 2026-02-24 162021" src="https://github.com/user-attachments/assets/285a7e92-26c5-420a-8632-ad53b2449cc3" />


### Predictions 




<img width="1325" height="781" alt="Screenshot 2026-02-24 162119" src="https://github.com/user-attachments/assets/bbaef90c-0d2a-4f8c-bee1-cff2bb443d8d" />


## Result
The Recurrent Neural Network (LSTM) model was successfully implemented and trained using historical Google stock price data. The training loss decreased gradually over epochs, indicating effective learning of temporal patterns in the dataset. When tested on unseen data, the predicted stock prices closely followed the actual stock prices, demonstrating that the model was able to capture trends and dependencies in the time series data. Thus, the experiment confirms that RNN-based models are suitable and effective for stock price prediction tasks.


