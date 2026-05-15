# Bangkok Air Quality Forecasting

## Overview

This project predicts next-day PM2.5 levels in Bangkok using LSTM-based time-series models trained on public historical air quality data.

The model uses the previous 7 daily records of:

- PM2.5
- PM10

to predict the next-day PM2.5 value.

This project compares:

- Naive Baseline: uses the latest PM2.5 value as the next-day prediction
- Vanilla LSTM: `LSTM(32) → Dropout → Dense(1)`
- Improved LSTM: `LSTM(32) → Dropout → Dense(16, tanh) → Dense(1)`

The improved model adds a Dense hidden layer with 16 units and `tanh` activation to learn an additional nonlinear transformation after the LSTM representation.

## Dataset

Dataset source: Kaggle - AQI Jakarta, Hanoi, Bangkok, Kuala Lumpur  
https://www.kaggle.com/datasets/yasirabd/aqi-jakarta-hanoi-bangkok-kuala-lumpur/data

This project uses the Bangkok air quality data.

After removing rows with missing PM2.5 or PM10 values, the usable data period is:

- Start date: 2016-07-30
- End date: 2020-12-06
- Original rows: 2,528
- Cleaned rows: 1,578

## Results

| Model | MAE | RMSE |
|---|---:|---:|
| Naive Baseline | 10.032 | 13.988 |
| Vanilla LSTM | 5.139 | 6.784 |
| Improved LSTM Dense-Tanh | 4.718 | 6.337 |

The improved LSTM reduced MAE by **8.15%** and RMSE by **6.60%** compared with the vanilla LSTM.

## Confidence Interval Analysis

Bootstrap 95% confidence intervals were computed to evaluate whether the improved model consistently reduces prediction error.

| Model | MAE 95% CI | RMSE 95% CI |
|---|---:|---:|
| Vanilla LSTM | [4.650, 5.641] | [6.136, 7.423] |
| Improved LSTM | [4.257, 5.200] | [5.686, 6.989] |

| Improvement | Mean | 95% CI |
|---|---:|---:|
| MAE Improvement | 8.15% | [2.81%, 13.29%] |
| RMSE Improvement | 6.60% | [1.95%, 11.34%] |

Since both improvement confidence intervals are positive, the improved LSTM shows a meaningful reduction in error compared with the vanilla LSTM.

## How to Run

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run confidence interval analysis
python confidence_interval.py
```