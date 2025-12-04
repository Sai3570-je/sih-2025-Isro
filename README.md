# 🛰️ ISRO Satellite Position Error Forecasting

**SIH 2025 Challenge: NAVIC Satellite Position & Clock Bias Error Prediction**

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()
[![Accuracy](https://img.shields.io/badge/MAE-0.327m-blue)]()
[![Gaussianity](https://img.shields.io/badge/Gaussian-100%25-success)]()

---

## 📋 Overview

This project implements a **Wavelet-Kalman Hybrid Time-Series Forecasting System** to predict NAVIC satellite position errors (X, Y, Z) and clock bias errors 24 hours in advance.

### Key Achievements

✅ **2.8× Better Accuracy** than baseline models (MAE: 0.327m vs 0.910m)  
✅ **100% Gaussian Residuals** (only model to achieve this)  
✅ **95% Confidence Intervals** with overfitting prevention  
✅ **96 Time-Slot Forecasts** (15-minute intervals for entire day)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Sai3570-je/sih-2025-Isro.git
cd sih-2025-Isro

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy pandas scipy pywt scikit-learn matplotlib seaborn
```

### Run Pipeline

```bash
# 1. Preprocess data (merge satellites)
python src/preprocess.py

# 2. Group by time of day (create 60 time slots)
python src/group_by_time.py
python src/split_by_time_of_day.py

# 3. Train Wavelet-Kalman models
python src/train_timeslots.py

# 4. Interpolate to 96 intervals (15-min grid)
python src/interpolate_15min.py

# 5. Visualize confidence intervals (optional)
python src/visualize_confidence.py
```

---

## 📁 Project Structure

```
sih-2025-Isro/
│
├── data/                          # Training data (Days 1-7)
│   ├── train_data_day1.csv
│   └── ... (days 2-7)
│
├── src/                           # Source code
│   ├── preprocess.py              # Data merging
│   ├── group_by_time.py           # Time-of-day grouping
│   ├── split_by_time_of_day.py    # Create time-slot files
│   ├── train_timeslots.py         # Main training (Wavelet-Kalman)
│   ├── kalman_filter.py           # Kalman implementation
│   ├── interpolate_15min.py       # Spline interpolation
│   ├── visualize_confidence.py    # CI visualizations
│   ├── compute_metrics.py         # Evaluation metrics
│   └── utils.py                   # Helper functions
│
├── outputs/                       # Results
│   ├── day8_forecast_15min.csv    # 🎯 FINAL OUTPUT (96 predictions)
│   ├── day8_forecast_timeslots.csv # 60 time-slot predictions
│   ├── wavelet_kalman_model.pkl   # Trained models
│   ├── time_series_groups/        # 60 time-slot training files
│   └── plots/                     # Visualizations
│       ├── predictions_with_confidence.png
│       ├── uncertainty_distribution.png
│       └── confidence_width_over_time.png
│
├── analysis/                      # Documentation
│   └── notebook.ipynb             # Complete analysis
│
├── README.md                      # This file
└── SUMMARY.md                     # Comprehensive documentation
```

---

## 🔬 Model Architecture

### Wavelet-Kalman Hybrid

```
Input: 7 days of satellite data
         ↓
[1] Time-of-Day Grouping (60 slots)
         ↓
[2] Wavelet Denoising (db4, level 2)
         ↓
[3] Kalman Filtering (state + trend)
         ↓
[4] One-Step-Ahead Forecast
         ↓
[5] Confidence Intervals (95%)
         ↓
[6] Spline Interpolation (96 intervals)
         ↓
Output: Day 8 predictions + uncertainty
```

### Key Components

**Wavelet Denoising:**
- Type: Daubechies 4 (db4)
- Levels: Adaptive (1 for <8 samples, 2 otherwise)
- Threshold: Soft thresholding with VisuShrink

**Kalman Filter:**
- State: [value, trend]
- Process Noise: Q = diag([0.1, 0.05])
- Measurement Noise: R from data variance
- Forecast: Damped trend (50% of last change)

**Uncertainty Quantification:**
- Dual sources: Kalman covariance + data variability
- Sample penalties: 1.5× for <5 samples, 1.2× for <7
- 95% CI = prediction ± 1.96σ

---

## 📊 Performance Results

### Model Comparison

| Model | MAE (m) | RMSE (m) | Gaussianity |
|-------|---------|----------|-------------|
| **Wavelet-Kalman** | **0.327** | **0.476** | **100%** ✅ |
| Kalman-Only | 0.910 | 1.319 | 0% ❌ |
| Linear Trend | 1.271 | 1.844 | 0% ❌ |
| Simple Mean | 1.295 | 1.877 | 0% ❌ |
| ARIMA | 1.380 | 2.000 | 0% ❌ |

### Uncertainty Statistics

| Error Type | Mean Uncertainty | CI Width (95%) | Status |
|------------|------------------|----------------|--------|
| X Position | 0.98 m | 3.83 m | ✅ Healthy |
| Y Position | 1.36 m | 5.35 m | ⚠️ Borderline |
| Z Position | 1.09 m | 4.28 m | ⚠️ Borderline |
| Clock Bias | 0.74 m | 2.88 m | ✅ Healthy |

---

## 📈 Output Files

### Primary Output: `day8_forecast_15min.csv`

**Columns:**
- `interval_start`, `interval_end` - Time boundaries
- `x_error_day8`, `y_error_day8`, `z_error_day8`, `clock_error_day8` - Predictions
- `x_uncertainty`, `y_uncertainty`, ... - Uncertainty estimates
- `x_conf_lower`, `x_conf_upper`, ... - 95% confidence intervals

**Coverage:** 96 time slots (00:00, 00:15, ..., 23:45)

---

## 🎯 Key Features

### Overfitting Prevention (5 Mechanisms)

1. **Dual Uncertainty Sources** - Combines Kalman + data variability
2. **Sample Size Penalties** - Wider intervals for sparse data
3. **Doubled Process Noise** - Accounts for forecast uncertainty
4. **Trend Damping** - Prevents wild extrapolations (50% damping)
5. **Adaptive Wavelet Levels** - Shallow decomposition for short sequences

### Statistical Rigor

✅ Shapiro-Wilk test (100% Gaussian residuals)  
✅ Kurtosis analysis (normal tail behavior)  
✅ Confidence interval validation (95% coverage)  
✅ Overfitting checks (0% low-uncertainty slots)

---

## 📖 Documentation

**Complete Documentation:** See [`SUMMARY.md`](SUMMARY.md)

**Analysis Notebook:** [`analysis/notebook.ipynb`](analysis/notebook.ipynb)

**Topics Covered:**
- Model architecture details
- Hyperparameter selection
- Baseline comparisons
- Confidence interval methodology
- ConTra paper analysis
- Deployment recommendations

---

## 🔧 Dependencies

**Core:**
- Python 3.10+
- numpy, pandas, scipy
- PyWavelets (pywt)
- scikit-learn

**Visualization:**
- matplotlib, seaborn

---

## 📞 Contact

**Repository:** [github.com/Sai3570-je/sih-2025-Isro](https://github.com/Sai3570-je/sih-2025-Isro)

**Team:** SIH 2025 ISRO Challenge

---

## 📝 License

Developed for **SIH 2025 - ISRO Challenge**  
Research and educational purposes

---

## 🏆 Status

✅ **PRODUCTION-READY**

**Final Outputs:**
- `outputs/day8_forecast_15min.csv` - 96 predictions with CI
- `outputs/wavelet_kalman_model.pkl` - Trained models
- `outputs/plots/` - Confidence interval visualizations

🚀 **Ready for ISRO submission!**
