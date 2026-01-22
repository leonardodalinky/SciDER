# Quick Start Guide - Wine Quality ML Pipeline

## TL;DR

```bash
# Activate environment
source .venv/bin/activate

# Run quick test (30 seconds)
python main.py --quick-test

# Run full pipeline (3-5 minutes)
python main.py
```

## What's New in Revision 2

✅ **Enhanced Reliability**
- Automatic directory creation with error handling
- File verification after every save operation
- Comprehensive output verification (Step 9)
- Visual indicators (✓/✗) for all operations

✅ **Better Error Handling**
- Try-except blocks for all file I/O
- Clear, actionable error messages
- Graceful error reporting

✅ **Complete Visibility**
- File sizes reported for all outputs
- Verification of all generated files
- Clear success/failure indicators

## Command Reference

### Basic Commands

```bash
# Quick test (3 models, ~30 seconds)
python main.py --quick-test

# Full pipeline (6 models, ~3-5 minutes)
python main.py

# With 10-fold cross-validation (~5-8 minutes)
python main.py --cv-folds 10

# Red wine only
python main.py --wine-type red

# White wine only
python main.py --wine-type white

# Skip EDA plots
python main.py --skip-eda

# Custom test size (default: 0.2)
python main.py --test-size 0.3
```

### Combining Options

```bash
# Quick test with red wine only
python main.py --quick-test --wine-type red

# Full pipeline with 10-fold CV, no EDA
python main.py --cv-folds 10 --skip-eda

# Custom seed for reproducibility
python main.py --random-seed 123
```

## Output Verification

After running the pipeline, Step 9 automatically verifies all outputs:

```
Step 9: Verifying Output Files...

Verifying results/ directory:
  ✓ model_comparison.csv (986 bytes)
  ✓ detailed_results.json (1,847 bytes)
  ✓ model_comparison.png (237,254 bytes)
  ✓ confusion_matrices.png (105,418 bytes)
  ✓ roc_curves.png (203,914 bytes)
  ✓ training_time_comparison.png (94,299 bytes)
  ✓ quality_distribution.png (73,824 bytes)
  ✓ correlation_matrix.png (529,417 bytes)
  ✓ feature_distributions.png (424,075 bytes)
  ✓ quality_vs_features.png (527,268 bytes)

Verifying models/ directory:
  ✓ xgboost.joblib (337,546 bytes)

✓ All expected output files verified successfully!
```

## Expected Results

### Quick Test Mode (--quick-test)

**Models Trained:** Logistic Regression, Random Forest, XGBoost

**Typical Performance:**
- **Accuracy:** 74-81%
- **Best Model:** Usually XGBoost (~81%)
- **Execution Time:** ~30 seconds

**Generated Files:**
- `results/`: 10 files (CSV, JSON, PNGs)
- `models/`: 1 file (xgboost.joblib)

### Full Pipeline

**Models Trained:** All 6 models (LR, DT, RF, GB, XGBoost, SVM)

**Typical Performance:**
- **Accuracy Range:** 69-81%
- **Best Model:** Usually XGBoost (~81%)
- **Execution Time:** ~3-5 minutes

**Generated Files:**
- `results/`: 11+ files (includes learning curves)
- `models/`: 1 file (best model)

## Common Use Cases

### 1. Quick Verification
```bash
python main.py --quick-test
```
Use when: Verifying installation, testing changes, quick insights

### 2. Full Analysis
```bash
python main.py
```
Use when: Complete model comparison, publication-quality results

### 3. Robust Evaluation
```bash
python main.py --cv-folds 10
```
Use when: Need robust performance estimates, research purposes

### 4. Wine Type Comparison
```bash
python main.py --wine-type red > red_results.txt
python main.py --wine-type white > white_results.txt
python main.py --wine-type both > combined_results.txt
```
Use when: Comparing red vs white wine patterns

### 5. Clean Start
```bash
rm -rf results/* models/*
python main.py
```
Use when: Want to regenerate all outputs from scratch

## Troubleshooting

### Issue: Import errors
**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Verify installation
python test_installation.py
```

### Issue: Data not found
**Solution:**
```bash
# Check data files exist
ls data/

# Should see:
# winequality-red.csv
# winequality-white.csv
```

### Issue: Missing output files
**Solution:**
- Check Step 9 verification output
- Look for ✗ marks indicating failures
- Verify write permissions: `ls -la results/ models/`
- Re-run pipeline to regenerate

### Issue: Memory errors
**Solution:**
```bash
# Use quick test mode
python main.py --quick-test

# Or skip EDA
python main.py --skip-eda
```

## Reading Results

### Console Output
- **Step 1-4:** Setup and initialization
- **Step 5:** Training progress with time per model
- **Step 6:** Best model identification
- **Step 7:** Visualization generation
- **Step 8:** Model saving
- **Step 9:** Output verification ⭐ NEW

### Model Comparison Table
```
         model_name  accuracy  precision   recall  f1_score  roc_auc
            XGBoost  0.813846   0.834292 0.880923  0.856974 0.876983
      Random Forest  0.807692   0.823729 0.885784  0.853630 0.880911
Logistic Regression  0.739231   0.766520 0.845687  0.804159 0.805668
```

### Key Files
- `results/model_comparison.csv` - All metrics in CSV format
- `results/detailed_results.json` - Complete results including confusion matrices
- `results/model_comparison.png` - Visual comparison of all models
- `models/xgboost.joblib` - Trained best model (typically XGBoost)

## Performance Tips

1. **For Speed:** Use `--quick-test` (10x faster)
2. **For Accuracy:** Use `--cv-folds 10` (more robust estimates)
3. **For Red Wine Only:** Use `--wine-type red` (smaller dataset)
4. **Skip Plots:** Use `--skip-eda` (saves ~5 seconds)

## Next Steps

After running the pipeline:

1. **View Results:**
   ```bash
   open results/  # macOS
   xdg-open results/  # Linux
   explorer results\  # Windows
   ```

2. **Analyze CSV:**
   ```bash
   cat results/model_comparison.csv
   ```

3. **Load Best Model:**
   ```python
   import joblib
   model = joblib.load('models/xgboost.joblib')
   # Now you can use model.predict(X_new)
   ```

4. **Compare Results:**
   - View `results/model_comparison.png` for visual comparison
   - Check `results/confusion_matrices.png` for error analysis
   - Examine `results/roc_curves.png` for threshold analysis

## File Size Reference

**Normal file sizes:**
- CSV files: ~1-2 KB
- JSON files: ~2-3 KB
- PNG plots: 50-500 KB
- Model files: 200-500 KB

**Warning signs:**
- File sizes of 0 bytes (indicates failed save)
- Missing files (check ✗ marks in Step 9)
- Very large files (>10 MB, unexpected)

## Support

For issues or questions:
1. Check `CHANGELOG.md` for recent changes
2. Read `README.md` for detailed documentation
3. Review `REVISION_2_SUMMARY.md` for technical details
4. Verify installation with `python test_installation.py`

---

**Happy Machine Learning!** 🍷🤖
