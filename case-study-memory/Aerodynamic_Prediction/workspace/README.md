# Airfoil Self-Noise Neural Network Regression

This project implements a complete machine learning pipeline for predicting sound pressure levels from the Airfoil Self-Noise dataset using three different neural network architectures.

## Dataset

The Airfoil Self-Noise dataset contains 1,503 samples with the following features:
- **Frequency** (Hz)
- **Angle of attack** (degrees)
- **Chord length** (meters)
- **Free-stream velocity** (m/s)
- **Suction side displacement thickness** (meters)
- **Target**: Scaled sound pressure level (dB)

## Neural Network Models

The implementation compares three different neural network architectures:

### 1. Simple Feedforward Network
- **Architecture**: Input(5) → Dense(64, ReLU) → Output(1)
- **Description**: A simple 2-layer network serving as a baseline
- **Best for**: Quick training and simple patterns

### 2. Deep Network with Dropout
- **Architecture**: Input(5) → Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(32) → Output(1)
- **Description**: Deeper network with dropout regularization to prevent overfitting
- **Best for**: Complex patterns with regularization

### 3. Wide Network with Batch Normalization
- **Architecture**: Input(5) → Dense(256) → BatchNorm → Dense(128) → BatchNorm → Output(1)
- **Description**: Wide network with batch normalization for stable training
- **Best for**: Capturing diverse feature interactions

## Project Structure

```
workspace/
├── data/
│   └── airfoil_self_noise.dat    # Dataset file
├── airfoil_regression.py          # Main implementation
├── pyproject.toml                 # Project dependencies
├── README.md                      # This file
├── results.txt                    # Detailed results (generated)
└── training_results.png           # Visualization (generated)
```

## Setup

This project uses `uv` for Python environment management. The environment is already configured with the following dependencies:

- **numpy** >= 1.24.0
- **pandas** >= 2.0.0
- **scikit-learn** >= 1.3.0
- **torch** >= 2.0.0
- **matplotlib** >= 3.7.0

To sync the environment (if needed):
```bash
uv sync
```

## Usage

### Quick Demo (Fast Testing)

Run with reduced epochs (50 epochs) for quick testing:

```bash
uv run python airfoil_regression.py --demo
```

This will:
- Load and preprocess the data
- Train all three models with 50 epochs
- Evaluate on the test set
- Generate visualizations and results
- Complete in ~3-5 seconds

### Full Training (Production)

Run with full training (200 epochs) for best results:

```bash
uv run python airfoil_regression.py
```

This will:
- Train all three models with 200 epochs
- Produce more accurate predictions
- Take ~10-20 seconds to complete

**Note**: The demo mode uses only 50 epochs, which may not be sufficient for optimal performance. The full training with 200 epochs will produce significantly better results.

## Output Files

After running the script, the following files will be generated:

1. **training_results.png**: Contains two plots:
   - Training loss curves for all three models
   - RMSE comparison bar chart

2. **results.txt**: Detailed text report including:
   - Dataset statistics
   - Training configuration
   - Evaluation metrics for each model (MSE, RMSE, MAE, R²)
   - Training times
   - Best model identification

## Evaluation Metrics

The models are evaluated using the following metrics:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE (same units as target)
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **R² Score**: Coefficient of determination (1.0 = perfect predictions, <0 = worse than mean)

## Data Preprocessing

The pipeline includes:
1. **Data Loading**: Reading the .dat file with proper column naming
2. **Train/Test Split**: 80% training, 20% testing (stratified)
3. **Feature Scaling**: StandardScaler applied to both features and target
4. **Batch Processing**: Mini-batch training with batch size of 32

## Implementation Details

- **Framework**: PyTorch
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: MSE (Mean Squared Error)
- **Batch Size**: 32
- **Random Seed**: 42 (for reproducibility)
- **Device**: Automatically detects CUDA if available, otherwise uses CPU

## Expected Performance

With full training (200 epochs), you should expect:
- **RMSE**: ~2-5 dB (good performance)
- **R² Score**: ~0.85-0.95 (high correlation)
- **Training Time**: 10-20 seconds per model on CPU

The demo mode (50 epochs) will show higher errors as models haven't fully converged.

## Extending the Code

To modify or extend the implementation:

1. **Add new models**: Create a new class inheriting from `nn.Module` in the script
2. **Adjust hyperparameters**: Modify `epochs`, `learning_rate`, or `batch_size` in the `main()` function
3. **Change preprocessing**: Modify the `load_and_preprocess_data()` function
4. **Add more metrics**: Update the `evaluate_model()` function

## Troubleshooting

### Issue: Poor model performance (negative R²)
- **Solution**: Run without `--demo` flag to train for 200 epochs instead of 50

### Issue: Training too slow
- **Solution**: Reduce number of epochs or use GPU if available

### Issue: Out of memory
- **Solution**: Reduce batch size in the DataLoader creation

## Next Steps

After reviewing the code changes, you can:

1. **Run full experiments**:
   ```bash
   uv run python airfoil_regression.py
   ```

2. **Experiment with hyperparameters**: Modify the script to test different:
   - Learning rates
   - Network architectures
   - Batch sizes
   - Dropout rates

3. **Analyze results**: Review `results.txt` and `training_results.png` to compare models

4. **Deploy best model**: Save and use the best-performing model for predictions

## References

- Dataset: UCI Machine Learning Repository - Airfoil Self-Noise Dataset
- Framework: PyTorch (https://pytorch.org/)
- Preprocessing: scikit-learn (https://scikit-learn.org/)
