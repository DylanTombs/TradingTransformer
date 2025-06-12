import numpy as np
import pandas as pd
from Transformer import Transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === Load and prepare dataset ===
def load_data(filepath, window=30):
    df = pd.read_csv(filepath)

    features = df[['open', 'high', 'low', 'close', 'volume', 'ema', 'rsi', 'macd']].values
    target = df['future_return_3d'].values

    X, y = [], []
    for i in range(len(df) - window):
        X.append(features[i:i + window])
        y.append(target[i + window])

    X = np.array(X)  # (batch, seq_len, num_features)
    y = np.array(y).reshape(-1, 1)  # (batch, 1)
    print(X.shape)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# === Train function ===
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    for epoch in range(epochs):
        # Forward pass
        x_encoded = model.forward(X_train)  # (batch, seq_len, num_features)
        predictions = x_encoded  # final output is already regression logits

        # Compute loss
        loss = np.mean((predictions - y_train) ** 2)

        # === Gradient Descent ===
        grad_output = 2 * (predictions - y_train) / y_train.shape[0]  # dL/dy

        # Backprop through final Linear layer (regression head)
        last_hidden = model.encoder_layers[-1].forward(model.pe.forward(model.input_proj.forward(X_train)))

        # Reshape for matrix multiplication
        last_hidden_flat = last_hidden.reshape(-1, last_hidden.shape[-1])  # (batch*seq_len, d_model)
        grad_output_flat = grad_output.reshape(-1, 1)  # (batch*seq_len, 1)
        
        # Ensure shapes are compatible
        if len(grad_output_flat) != len(last_hidden_flat):
            # If sequence length > 1, we need to repeat gradients for each timestep
            grad_output_flat = np.repeat(grad_output, last_hidden.shape[1], axis=0).reshape(-1, 1)

        # Compute gradients
        dW = np.matmul(last_hidden_flat.T, grad_output_flat)  # (d_model, 1)
        db = np.sum(grad_output_flat, axis=0)  # (1,)

        # Update weights
        model.output_layer.weight -= lr * dW.reshape(model.output_layer.weight.shape)
        model.output_layer.bias -= lr * db

        # Validation loss
        val_pred = model.forward(X_val)
        val_loss = np.mean((val_pred - y_val) ** 2)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss:.6f} - Val Loss: {val_loss:.6f}")


# === Main ===
if __name__ == '__main__':
    FILEPATH = 'TrainingData.csv'
    X_train, X_val, y_train, y_val = load_data(FILEPATH)

    transformer = Transformer(
        num_features=8,     # [open, high, low, close, volume, ema, rsi, macd]
        d_model=64,
        N=2,
        h=4,
        d_ff=128,
        dropout=0.1
    )

    train_model(transformer, X_train, y_train, X_val, y_val, epochs=20, lr=0.0001)
