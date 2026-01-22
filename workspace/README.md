# Wine Quality Machine Learning Project

A comprehensive machine learning pipeline for predicting wine quality based on physicochemical properties. This project implements multiple ML algorithms, provides extensive data analysis, and generates detailed visualizations for model comparison.

## Overview

This project uses the Wine Quality dataset (red and white wines) to predict wine quality using various machine learning algorithms. The pipeline includes:

- **Data Loading & Preprocessing**: Handles semicolon-delimited CSV files, feature scaling, and train/test splitting
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis and visualizations
- **Multiple ML Models**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and SVM
- **Model Evaluation**: Cross-validation, multiple metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- **Visualization**: Model comparison, confusion matrices, ROC curves, learning curves, and feature importance

## Project Structure

```
workspace/
├── data/                           # Wine quality datasets
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── src/
│   └── wine_quality_ml/            # Main package
│       ├── __init__.py
│       ├── data_loader.py          # Data loading and preprocessing
│       ├── eda.py                  # Exploratory data analysis
│       ├── models.py               # ML model definitions
│       ├── trainer.py              # Model training and evaluation
│       └── visualizer.py           # Results visualization
├── models/                         # Saved trained models
├── results/                        # Visualizations and evaluation results
├── main.py                         # Main execution script
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

## Setup

### Prerequisites

- Python 3.9+
- `uv` package manager (recommended)

### Installation

1. **Create virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   uv pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
   ```

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python main.py
```

### Quick Test Mode

For rapid testing with a subset of models:

```bash
python main.py --quick-test
```

### Command-Line Options

```bash
python main.py [OPTIONS]
```

**Options:**

- `--wine-type {red,white,both}`: Type of wine to analyze (default: both)
- `--test-size FLOAT`: Test set proportion (default: 0.2)
- `--skip-eda`: Skip exploratory data analysis
- `--cv-folds INT`: Number of cross-validation folds (default: 5)
- `--random-seed INT`: Random seed for reproducibility (default: 42)
- `--quick-test`: Run quick test with limited models (Logistic Regression, Random Forest, XGBoost)

### Examples

**Analyze only red wine:**
```bash
python main.py --wine-type red
```

**Use 10-fold cross-validation:**
```bash
python main.py --cv-folds 10
```

**Skip EDA and run quick test:**
```bash
python main.py --skip-eda --quick-test
```

**Custom test size and random seed:**
```bash
python main.py --test-size 0.3 --random-seed 123
```

## Models Implemented

The pipeline includes the following machine learning algorithms:

1. **Logistic Regression**: Linear baseline model
2. **Decision Tree**: Simple tree-based classifier
3. **Random Forest**: Ensemble of decision trees
4. **Gradient Boosting**: Sequential boosting algorithm
5. **XGBoost**: Advanced gradient boosting
6. **Support Vector Machine (SVM)**: Kernel-based classifier

## Evaluation Metrics

Models are evaluated using:

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Cross-Validation**: K-fold CV for robustness
- **Training Time**: Computational efficiency

## Output

After running the pipeline, the following outputs are generated:

### Results Directory (`results/`)

**EDA Visualizations:**
- `quality_distribution.png`: Distribution of quality scores
- `correlation_matrix.png`: Feature correlation heatmap
- `feature_distributions.png`: Individual feature distributions
- `quality_vs_features.png`: Quality vs top correlated features

**Model Evaluation:**
- `model_comparison.png`: Comparison of all models across metrics
- `confusion_matrices.png`: Confusion matrices for all models
- `roc_curves.png`: ROC curves comparison
- `training_time_comparison.png`: Training time comparison
- `feature_importance_*.png`: Feature importance for tree-based models
- `learning_curve_*.png`: Learning curves for best model

**Data Files:**
- `model_comparison.csv`: Detailed metrics for all models
- `detailed_results.json`: Complete evaluation results in JSON format

### Models Directory (`models/`)

- `{best_model_name}.joblib`: Saved best-performing model

## Dataset Information

### Wine Quality Dataset

The dataset contains physicochemical properties of Portuguese "Vinho Verde" wines:

**Features (11 input variables):**
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

**Target Variable:**
- **Quality**: Score between 0-10 (converted to binary: good quality >= 6)

**Dataset Sizes:**
- Red wine: ~1,599 samples
- White wine: ~4,898 samples
- Combined: ~6,497 samples

### Data Preprocessing

The pipeline automatically handles:
- Semicolon-delimited CSV parsing
- Missing value removal
- Feature scaling (StandardScaler)
- Binary classification conversion (quality >= 6)
- Stratified train/test splitting
- Wine type encoding (if using both types)

## Best Practices

### For Quick Testing

Use `--quick-test` to run only 3 models (Logistic Regression, Random Forest, XGBoost), which significantly reduces execution time while still providing good insights.

### For Production

Run the full pipeline without `--quick-test` to evaluate all 6 models and generate comprehensive visualizations including learning curves.

### For Reproducibility

Always specify `--random-seed` with a fixed value to ensure reproducible results across runs.

### For Different Wine Types

Analyze red and white wines separately to understand type-specific patterns, then compare with combined analysis.

## Technical Details

### Classification Strategy

The project uses **binary classification**:
- **High Quality**: quality score >= 6
- **Low Quality**: quality score < 6

This approach is chosen because:
1. Original quality scores are imbalanced
2. Binary classification is more practical for real-world applications
3. Better model performance and interpretability

### Feature Scaling

All features are scaled using `StandardScaler` to:
- Normalize feature ranges
- Improve convergence for gradient-based algorithms
- Ensure fair feature importance comparison

### Cross-Validation

Stratified K-fold cross-validation ensures:
- Balanced class distribution in each fold
- Robust performance estimates
- Detection of overfitting

## Extending the Project

### Adding New Models

Edit `src/wine_quality_ml/models.py` and add your model to the `_initialize_models` method:

```python
self.models['Your Model'] = YourModelClass(
    # your hyperparameters
)
```

### Custom Hyperparameters

Use the `ModelOptimizer` class in `models.py` to define parameter grids for hyperparameter tuning.

### Additional Visualizations

Extend `src/wine_quality_ml/visualizer.py` with custom plotting functions.

## Reliability & Robustness Features

This pipeline includes several features to ensure reliable execution and verifiable results:

### Automatic Directory Creation

All output directories (`results/`, `models/`) are automatically created with proper error handling. The pipeline ensures:
- Parent directories are created recursively if needed
- No errors occur if directories already exist
- Proper permissions are set for file operations

### File Verification

After execution, the pipeline automatically verifies all generated files:
- **Checkmarks (✓)**: Indicate successful file creation with file sizes
- **X marks (✗)**: Indicate missing or failed files
- File sizes are displayed to confirm non-empty outputs

### Error Handling

Robust error handling throughout:
- **IOError handling**: Catches and reports file I/O errors
- **Directory creation verification**: Confirms directories exist before writing
- **File existence checks**: Verifies files after saving operations
- **Graceful degradation**: Pipeline continues even if optional components fail

### Output Verification Step

The final step (Step 9) comprehensively verifies:
- All required result files (CSV, JSON, PNG)
- All model files (.joblib)
- Optional files (learning curves, EDA plots)
- Provides a clear summary of what was successfully created

Example output:
```
Step 9: Verifying Output Files...

Verifying results/ directory:
  ✓ model_comparison.csv (985 bytes)
  ✓ detailed_results.json (1,846 bytes)
  ✓ model_comparison.png (237,254 bytes)
  ...

✓ All expected output files verified successfully!
```

## Troubleshooting

### Import Errors

Ensure you're in the workspace directory and the virtual environment is activated.

### Memory Issues

Use `--quick-test` to reduce computational load, or increase system resources.

### Data Not Found

Verify that `data/winequality-red.csv` and `data/winequality-white.csv` exist in the workspace.

### Missing Output Files

If the verification step shows missing files:
1. Check the error messages during execution
2. Ensure sufficient disk space is available
3. Verify write permissions for `results/` and `models/` directories
4. Re-run the pipeline to regenerate missing outputs

## Performance Expectations

With default settings on a typical machine:

- **Quick Test Mode**: ~30 seconds - 1 minute
- **Full Pipeline**: ~2-5 minutes (depending on CPU)
- **With Learning Curves**: Additional 2-3 minutes

## Citation

If you use this project or the Wine Quality dataset, please cite:

```
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
```

## License

This project is provided for educational and research purposes.

## Contact

For questions or suggestions, please refer to the project repository or documentation.

---

**Happy Machine Learning!** 🍷🤖
