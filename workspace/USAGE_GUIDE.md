# Usage Guide - Wine Quality ML Pipeline

## Quick Start Guide

### 1. First Time Setup

```bash
# Navigate to the workspace
cd /Users/harrylu_mac/scievo/SciEvo/workspace

# Activate virtual environment
source .venv/bin/activate  # or on Windows: .venv\Scripts\activate

# Verify installation
python -c "import pandas, sklearn, xgboost; print('All dependencies installed!')"
```

### 2. Run Quick Test (Recommended First Run)

This runs a quick test with 3 models and takes ~30 seconds:

```bash
python main.py --quick-test
```

### 3. Run Full Pipeline

To run all 6 models with comprehensive analysis (~3-5 minutes):

```bash
python main.py
```

## Common Use Cases

### Analyze Specific Wine Type

**Red wine only:**
```bash
python main.py --wine-type red
```

**White wine only:**
```bash
python main.py --wine-type white
```

**Both types (default):**
```bash
python main.py --wine-type both
```

### Skip Exploratory Data Analysis

If you've already seen the EDA plots and want to focus on modeling:

```bash
python main.py --skip-eda
```

### Custom Train/Test Split

Default is 80/20 split. To use 70/30:

```bash
python main.py --test-size 0.3
```

### More Cross-Validation Folds

Default is 5-fold CV. To use 10-fold:

```bash
python main.py --cv-folds 10
```

### Reproducibility

Always use the same random seed:

```bash
python main.py --random-seed 42
```

## Expected Results

### Model Performance Rankings (Typical)

1. **XGBoost**: ~81-82% accuracy
2. **Random Forest**: ~80-81% accuracy
3. **Gradient Boosting**: ~80-81% accuracy
4. **SVM**: ~78-79% accuracy
5. **Decision Tree**: ~75-76% accuracy
6. **Logistic Regression**: ~73-75% accuracy

### Output Files

After running, check these directories:

**results/**
- EDA plots (4 images)
- Model comparison visualizations (7+ images)
- CSV and JSON files with detailed metrics

**models/**
- Trained model file (`.joblib` format)

## Interpreting Results

### Model Comparison Table

The console output shows a comparison table:

```
model_name         accuracy  precision  recall  f1_score  roc_auc  cv_mean  cv_std  training_time
XGBoost            0.8138    0.8343     0.8809  0.8570    0.8770   0.8084   0.0056  0.21
Random Forest      0.8077    0.8237     0.8858  0.8536    0.8809   0.8101   0.0120  0.09
Logistic Regr...   0.7392    0.7665     0.8457  0.8042    0.8057   0.7406   0.0052  0.98
```

**Key Metrics:**
- **Accuracy**: Overall correctness (higher is better)
- **Precision**: Of predicted high-quality wines, how many are actually high-quality
- **Recall**: Of all high-quality wines, how many did we find
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)
- **ROC-AUC**: Area under ROC curve (closer to 1.0 is better)
- **CV Mean/Std**: Cross-validation results (lower std = more stable)
- **Training Time**: Seconds to train (lower is faster)

### Classification Report

The detailed report shows per-class performance:

```
              precision    recall  f1-score   support
Low Quality       0.77      0.70      0.73       477
High Quality      0.83      0.88      0.86       823
```

- **Low Quality**: Wines with quality score < 6
- **High Quality**: Wines with quality score >= 6
- **Support**: Number of samples in test set

## Advanced Usage

### Loading a Saved Model

```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('models/xgboost.joblib')

# Prepare your data (must be scaled)
scaler = StandardScaler()
# ... fit scaler on your training data ...
X_new = scaler.transform(your_new_data)

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

### Using Individual Modules

```python
import sys
sys.path.insert(0, 'src')

from wine_quality_ml.data_loader import WineDataLoader
from wine_quality_ml.eda import WineEDA
from wine_quality_ml.models import WineQualityModels

# Load data
loader = WineDataLoader('data')
df = loader.load_data('both')

# Run EDA
eda = WineEDA('results')
eda.generate_full_report(df)

# Get models
models = WineQualityModels()
rf_model = models.get_model('Random Forest')
```

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:** Ensure virtual environment is activated:
```bash
source .venv/bin/activate
```

### Issue: "FileNotFoundError" for data files

**Solution:** Check that data files exist:
```bash
ls data/
# Should show: winequality-red.csv  winequality-white.csv
```

### Issue: Out of memory

**Solution:** Use quick test mode:
```bash
python main.py --quick-test
```

### Issue: Plots not showing

**Solution:** Plots are saved to `results/` directory, not displayed. Check:
```bash
ls results/*.png
```

## Performance Optimization

### Faster Execution

1. Use `--quick-test` (3 models instead of 6)
2. Use `--skip-eda` (skip visualization generation)
3. Reduce `--cv-folds` (e.g., `--cv-folds 3`)
4. Use smaller test size (e.g., `--test-size 0.1`)

**Fastest possible run:**
```bash
python main.py --quick-test --skip-eda --cv-folds 3 --test-size 0.1
```

### Better Accuracy

1. Use more cross-validation folds (`--cv-folds 10`)
2. Analyze wine types separately for type-specific models
3. Use the full pipeline (not quick-test)
4. Increase training data (lower test-size, e.g., `--test-size 0.15`)

## Next Steps After Running

1. **Review visualizations** in `results/` directory:
   - Check correlation matrix to understand feature relationships
   - Review confusion matrices to see error patterns
   - Analyze ROC curves for model comparison
   - Examine feature importance to understand key factors

2. **Analyze model performance**:
   - Compare accuracy vs training time trade-offs
   - Review cross-validation stability (cv_std)
   - Check classification report for per-class performance

3. **Experiment with different settings**:
   - Try different wine types
   - Adjust train/test splits
   - Compare results across multiple runs

4. **Use the best model**:
   - Load the saved model from `models/` directory
   - Apply it to new wine quality predictions
   - Integrate into your application

## Resources

- **Dataset Info**: See `README.md` for detailed dataset description
- **Code Documentation**: All modules have detailed docstrings
- **Paper Citation**: Cortez et al., Decision Support Systems, 2009

## Getting Help

If you encounter issues:

1. Check this guide for common problems
2. Review the `README.md` for detailed information
3. Examine the code comments in `src/wine_quality_ml/`
4. Verify all dependencies are installed correctly

---

**Happy analyzing!** 🍷📊
