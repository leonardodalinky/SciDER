"""
Airfoil Self-Noise Neural Network Regression

This script implements a complete machine learning pipeline for the Airfoil Self-Noise dataset:
- Data loading and preprocessing
- Train/test split
- Feature scaling
- Three different neural network architectures
- Model training and evaluation
- Comprehensive metrics reporting
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# Data Loading and Preprocessing
def load_and_preprocess_data(data_path="data/airfoil_self_noise.dat", test_size=0.2):
    """
    Load the Airfoil Self-Noise dataset and perform preprocessing.

    Dataset columns:
    1. Frequency (Hz)
    2. Angle of attack (degrees)
    3. Chord length (meters)
    4. Free-stream velocity (m/s)
    5. Suction side displacement thickness (meters)
    6. Scaled sound pressure level (dB) - TARGET

    Args:
        data_path: Path to the .dat file
        test_size: Proportion of data for test set

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    print(f"Loading data from {data_path}...")

    # Load data (tab/whitespace separated)
    column_names = [
        "frequency",
        "angle_of_attack",
        "chord_length",
        "free_stream_velocity",
        "suction_side_thickness",
        "sound_pressure_level",
    ]

    df = pd.read_csv(data_path, sep="\t", names=column_names, header=None)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print("\nDataset statistics:")
    print(df.describe())

    print("\nChecking for missing values:")
    print(df.isnull().sum())

    # Separate features and target
    X = df.drop("sound_pressure_level", axis=1).values
    y = df["sound_pressure_level"].values.reshape(-1, 1)

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Feature scaling (important for neural networks)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    print("\nData preprocessing completed!")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
        y_train,
        y_test,
    )


# Neural Network Model 1: Simple Feedforward Network
class SimpleNN(nn.Module):
    """
    Simple 2-layer feedforward neural network.
    Architecture: Input -> Hidden(64) -> Output
    """

    def __init__(self, input_size=5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Neural Network Model 2: Deep Network with Dropout
class DeepNN(nn.Module):
    """
    Deeper network with dropout for regularization.
    Architecture: Input -> Hidden(128) -> Hidden(64) -> Hidden(32) -> Output
    """

    def __init__(self, input_size=5, dropout_rate=0.2):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Neural Network Model 3: Wide Network with Batch Normalization
class WideNN(nn.Module):
    """
    Wide network with batch normalization.
    Architecture: Input -> Hidden(256) -> Hidden(128) -> Output
    """

    def __init__(self, input_size=5):
        super(WideNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# Training function
def train_model(model, train_loader, criterion, optimizer, device, epochs=100, verbose=True):
    """
    Train a neural network model.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU/GPU)
        epochs: Number of training epochs
        verbose: Whether to print training progress

    Returns:
        List of training losses
    """
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return train_losses


# Evaluation function
def evaluate_model(model, X_test, y_test_scaled, scaler_y, device, y_test_original):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained PyTorch model
        X_test: Test features (scaled)
        y_test_scaled: Test targets (scaled) - used for model input
        scaler_y: Scaler for inverse transforming predictions
        device: Device to evaluate on
        y_test_original: Test targets in original scale - used for metrics

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Calculate metrics using original scale targets
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "predictions": y_pred}


def main(demo_mode=False):
    """
    Main function to run the complete pipeline.

    Args:
        demo_mode: If True, uses fewer epochs for quick testing
    """
    print("=" * 70)
    print("AIRFOIL SELF-NOISE NEURAL NETWORK REGRESSION")
    print("=" * 70)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load and preprocess data
    print("\n" + "=" * 70)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
        y_train_original,
        y_test_original,
    ) = load_and_preprocess_data()

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model configuration
    input_size = X_train.shape[1]
    epochs = 50 if demo_mode else 200  # Reduced epochs for demo
    learning_rate = 0.001

    print(f"\nTraining configuration:")
    print(f"- Epochs: {epochs}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Batch size: 32")

    # Dictionary to store results
    results = {}

    # Model 1: Simple Feedforward Network
    print("\n" + "=" * 70)
    print("MODEL 1: SIMPLE FEEDFORWARD NETWORK")
    print("=" * 70)
    print("Architecture: Input(5) -> Dense(64, ReLU) -> Output(1)")

    model1 = SimpleNN(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)

    print("\nTraining Model 1...")
    start_time = time.time()
    train_losses1 = train_model(model1, train_loader, criterion, optimizer1, device, epochs)
    train_time1 = time.time() - start_time
    print(f"Training completed in {train_time1:.2f} seconds")

    print("\nEvaluating Model 1...")
    results["Model 1 - Simple NN"] = evaluate_model(
        model1, X_test, y_test_scaled, scaler_y, device, y_test_original
    )
    results["Model 1 - Simple NN"]["train_time"] = train_time1
    results["Model 1 - Simple NN"]["train_losses"] = train_losses1

    # Model 2: Deep Network with Dropout
    print("\n" + "=" * 70)
    print("MODEL 2: DEEP NETWORK WITH DROPOUT")
    print("=" * 70)
    print(
        "Architecture: Input(5) -> Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(32) -> Output(1)"
    )

    model2 = DeepNN(input_size).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

    print("\nTraining Model 2...")
    start_time = time.time()
    train_losses2 = train_model(model2, train_loader, criterion, optimizer2, device, epochs)
    train_time2 = time.time() - start_time
    print(f"Training completed in {train_time2:.2f} seconds")

    print("\nEvaluating Model 2...")
    results["Model 2 - Deep NN"] = evaluate_model(
        model2, X_test, y_test_scaled, scaler_y, device, y_test_original
    )
    results["Model 2 - Deep NN"]["train_time"] = train_time2
    results["Model 2 - Deep NN"]["train_losses"] = train_losses2

    # Model 3: Wide Network with Batch Normalization
    print("\n" + "=" * 70)
    print("MODEL 3: WIDE NETWORK WITH BATCH NORMALIZATION")
    print("=" * 70)
    print(
        "Architecture: Input(5) -> Dense(256) -> BatchNorm -> Dense(128) -> BatchNorm -> Output(1)"
    )

    model3 = WideNN(input_size).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)

    print("\nTraining Model 3...")
    start_time = time.time()
    train_losses3 = train_model(model3, train_loader, criterion, optimizer3, device, epochs)
    train_time3 = time.time() - start_time
    print(f"Training completed in {train_time3:.2f} seconds")

    print("\nEvaluating Model 3...")
    results["Model 3 - Wide NN"] = evaluate_model(
        model3, X_test, y_test_scaled, scaler_y, device, y_test_original
    )
    results["Model 3 - Wide NN"]["train_time"] = train_time3
    results["Model 3 - Wide NN"]["train_losses"] = train_losses3

    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS ON TEST SET")
    print("=" * 70)

    # Create comparison table
    print(
        "\n{:<25} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "Model", "MSE", "RMSE", "MAE", "R²", "Train Time"
        )
    )
    print("-" * 97)

    for model_name, metrics in results.items():
        print(
            "{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.2f}s".format(
                model_name,
                metrics["MSE"],
                metrics["RMSE"],
                metrics["MAE"],
                metrics["R2"],
                metrics["train_time"],
            )
        )

    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]["RMSE"])
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model[0]} (Lowest RMSE: {best_model[1]['RMSE']:.4f})")
    print("=" * 70)

    # Plot training curves with robust error handling
    print("\nGenerating training curves plot...")
    plot_filename = "training_results.png"

    try:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses1, label="Simple NN", alpha=0.7)
        plt.plot(train_losses2, label="Deep NN", alpha=0.7)
        plt.plot(train_losses3, label="Wide NN", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss (MSE)")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        models = list(results.keys())
        rmse_values = [results[m]["RMSE"] for m in models]
        colors = ["#3498db", "#e74c3c", "#2ecc71"]

        bars = plt.bar(range(len(models)), rmse_values, color=colors, alpha=0.7)
        plt.xlabel("Model")
        plt.ylabel("RMSE")
        plt.title("Test Set RMSE Comparison")
        plt.xticks(range(len(models)), ["Simple NN", "Deep NN", "Wide NN"], rotation=15, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")

        # Verify file was created successfully
        import os

        if os.path.exists(plot_filename) and os.path.getsize(plot_filename) > 0:
            print(f"✓ Plot saved successfully as '{plot_filename}'")
            print(f"  File size: {os.path.getsize(plot_filename)} bytes")
        else:
            raise IOError(f"Plot file '{plot_filename}' was not created or is empty")

    except Exception as e:
        print(f"✗ ERROR: Failed to generate plot '{plot_filename}'")
        print(f"  Error details: {str(e)}")
        raise

    # Save detailed results with robust error handling
    print("\nSaving detailed results to 'results.txt'...")
    results_filename = "results.txt"

    try:
        with open(results_filename, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("AIRFOIL SELF-NOISE NEURAL NETWORK REGRESSION - DETAILED RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write("Dataset Information:\n")
            f.write(f"- Total samples: {len(X_train) + len(X_test)}\n")
            f.write(f"- Training samples: {len(X_train)}\n")
            f.write(f"- Test samples: {len(X_test)}\n")
            f.write(f"- Features: {input_size}\n")
            f.write(f"- Target: Sound Pressure Level (dB)\n\n")

            f.write("Training Configuration:\n")
            f.write(f"- Epochs: {epochs}\n")
            f.write(f"- Learning rate: {learning_rate}\n")
            f.write(f"- Batch size: 32\n")
            f.write(f"- Optimizer: Adam\n")
            f.write(f"- Loss function: MSE\n\n")

            f.write("=" * 70 + "\n")
            f.write("Model Comparison\n")
            f.write("=" * 70 + "\n\n")

            for model_name, metrics in results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  MSE:        {metrics['MSE']:.6f}\n")
                f.write(f"  RMSE:       {metrics['RMSE']:.6f}\n")
                f.write(f"  MAE:        {metrics['MAE']:.6f}\n")
                f.write(f"  R² Score:   {metrics['R2']:.6f}\n")
                f.write(f"  Train Time: {metrics['train_time']:.2f}s\n\n")

            f.write("=" * 70 + "\n")
            f.write(f"Best Model: {best_model[0]}\n")
            f.write(f"Best RMSE: {best_model[1]['RMSE']:.6f}\n")
            f.write("=" * 70 + "\n")

        # Verify file was written successfully
        import os

        if os.path.exists(results_filename) and os.path.getsize(results_filename) > 0:
            print(f"✓ Results saved successfully to '{results_filename}'")
            print(f"  File size: {os.path.getsize(results_filename)} bytes")
        else:
            raise IOError(f"File '{results_filename}' was not created or is empty")

    except Exception as e:
        print(f"✗ ERROR: Failed to save results to '{results_filename}'")
        print(f"  Error details: {str(e)}")
        raise

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Final verification of output files
    import os

    print("\nOutput files verification:")
    output_files = [
        ("training_results.png", "Visualization of training curves and RMSE comparison"),
        ("results.txt", "Detailed results and metrics"),
    ]

    all_files_exist = True
    for filename, description in output_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✓ {filename} ({size:,} bytes) - {description}")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            all_files_exist = False

    if not all_files_exist:
        print("\n⚠ WARNING: Some output files are missing!")
    else:
        print("\n✓ All output files generated successfully!")

    return results


if __name__ == "__main__":
    import sys

    # Check for demo mode flag
    demo_mode = "--demo" in sys.argv

    if demo_mode:
        print("Running in DEMO mode (reduced epochs for quick testing)...\n")

    results = main(demo_mode=demo_mode)
