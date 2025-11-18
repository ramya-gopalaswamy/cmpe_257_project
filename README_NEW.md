# CMPE 257 Project: Stock Price Prediction

A machine learning project to predict stock prices using historical stock data with sliding window features and walk-forward validation.

## Project Overview

This project processes historical stock market data and prepares it for ML model training:
- **Raw Data**: Stock prices, fundamentals, and securities metadata from Kaggle
- **Processing**: Creates sliding windows (2-5 days) of features for each stock
- **Validation**: Walk-forward cross-validation (5 folds) for time-series data
- **Normalization**: StandardScaler pipeline for feature normalization
- **Goal**: Train ML models to predict next-day closing price

## Data Structure

```
data/
├── raw/                          # Kaggle CSV files (uploaded to git)
│   ├── prices.csv
│   ├── prices-split-adjusted.csv
│   ├── fundamentals.csv
│   └── securities.csv
├── naive_processed/              # Simple processing (ChatGPT baseline)
│   └── window_*.csv
├── full_processed/               # Full walk-forward validation
│   ├── X_train_window_*_fold_*.csv
│   ├── X_test_window_*_fold_*.csv
│   ├── X_eval_window_*.csv
│   ├── y_train_window_*_fold_*.csv
│   ├── y_test_window_*_fold_*.csv
│   └── y_eval_window_*.csv
└── normalized/                   # Normalized data (StandardScaler)
    └── window_*/
        ├── fold_*/
        │   ├── X_train.csv, X_test.csv, y_train.csv, y_test.csv
        │   └── scaler_pipeline.pkl
        ├── X_eval.csv
        └── y_eval.csv
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd cmpe_257_project
```

### 2. Set up Python Environment through conda or pip


### 3. Install Dependencies

Install required packages via conda:

```bash
conda install numpy pandas python-dateutil pytz six tzdata scikit-learn matplotlib -y
```

Or use pip if you prefer:

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, numpy, sklearn; print('All packages imported successfully!')"
```

**Note:** If you see `ModuleNotFoundError`, make sure conda environment is activated:
```bash
unalias python  # Remove any python aliases
conda activate cmpe257
```

## Running the Pipeline

Make all scripts executable:

```bash
chmod +x scripts/*.py
```

### Step 1: Explore Raw Data

```bash
python scripts/initial_data_exploration.py
```

This prints dataset info (columns, dtypes, missing values) for all CSV files in `data/raw/`.

### Step 2: Process Full Data (with Walk-Forward Validation)

```bash
python scripts/process_data_full.py
```

**What it does:**
- Loads raw prices data
- Splits into train/eval by date (80/20)
- Creates sliding windows (2-5 days) for each company
- Applies walk-forward validation (5 folds)
- Generates features: open, close, low, high, volume at each time step
- Saves to `data/full_processed/`

**Duration:** ~5-10 minutes depending on machine

**Output:** Separate train/test/eval sets for each window size and fold

### Step 3: Build Normalization Pipeline

```bash
python scripts/build_pipeline.py
```

**What it does:**
- Loads processed data from `data/full_processed/`
- Creates StandardScaler for each fold
- Fits scaler on training data
- Normalizes train/test/eval sets
- Saves normalized data to `data/normalized/`
- Saves fitted scalers (`scaler_pipeline.pkl`) for later use

**Duration:** ~2-3 minutes

**Output:** Normalized datasets ready for model training

### Step 4: Explore Processed Data

```bash
python scripts/full_data_exploration.py
```

This prints statistics on the processed/normalized data and generates visualization plots.

### Optional: Process Naive Data (Quick Baseline)

```bash
python scripts/process_data_naive.py
```

This creates a simple processed dataset (without walk-forward validation) for quick experiments and comparison.

---

## Quick Start (All at Once)

```bash
chmod +x scripts/*.py
python scripts/initial_data_exploration.py
python scripts/process_data_full.py
python scripts/build_pipeline.py
python scripts/full_data_exploration.py
```

---

## Project Structure

```
scripts/
├── initial_data_exploration.py   # Explore raw CSV files
├── process_data_full.py          # Full processing with walk-forward validation
├── process_data_naive.py         # Simple baseline processing
├── full_data_exploration.py      # Explore processed data
└── build_pipeline.py             # Normalize data with StandardScaler

data/
├── raw/                          # Raw Kaggle data
├── naive_processed/              # Simple processed data
├── full_processed/               # Walk-forward validation data
└── normalized/                   # Normalized data for model training

experiments/                       # Model training scripts (to be added)
reports/                          # Results and analysis
```

## Next Steps: Model Training

Once normalized data is ready (`data/normalized/`), you can:

1. **Train baseline models** (Linear Regression, Random Forest, etc.)
2. **Use cross-validation** to find optimal hyperparameters
3. **Generate graphs** showing:
   - Parameter tuning results
   - Model performance on eval set
   - Predictions vs actual prices
4. **Compare models** and document results

Use the normalized data structure:
```python
import pandas as pd
from pathlib import Path

window_size = 3
fold = 0

X_train = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/X_train.csv")
y_train = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/y_train.csv")
X_test = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/X_test.csv")
y_test = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/y_test.csv")
X_eval = pd.read_csv(f"data/normalized/window_{window_size}/X_eval.csv")
y_eval = pd.read_csv(f"data/normalized/window_{window_size}/y_eval.csv")

# Your model training here...
```

## Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - ML preprocessing and models
- `matplotlib` - Plotting and visualization
- `python-dateutil`, `pytz`, `tzdata` - Date/time handling

See `requirements.txt` for specific versions.

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'pandas'`
- **Solution:** Ensure conda environment is activated and packages are installed
  ```bash
  conda activate cmpe257
  conda install pandas -y
  ```

**Issue:** Scripts take too long
- **Solution:** This is normal for large datasets. Process data once, then reuse normalized sets.

**Issue:** Out of memory errors
- **Solution:** The full dataset is large. If needed, modify scripts to process in batches or reduce data sample.

## Authors & Notes
- Cale Payson, Sonali Lonkar
- Repository: CMPE 257 Course Project
- Data source: Kaggle
- Processing: Custom walk-forward validation pipeline
- Framework: scikit-learn for preprocessing and scaling
