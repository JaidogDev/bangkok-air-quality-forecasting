# Bangkok Air Quality Forecasting

## Overview

This project predicts next-day PM2.5 levels in Bangkok using an LSTM model trained on public historical air quality data.

The model uses the previous 7 records of:

- PM2.5
- PM10

to predict the next PM2.5 value.

The best model is selected using the lowest validation loss and evaluated using MAE and RMSE.

## Dataset

Dataset source:  
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
| LSTM | 5.139 | 6.784 |
| Naive Baseline | 10.032 | 13.988 |

The LSTM model outperformed the naive baseline, reducing MAE from 10.032 to 5.139.

## How to Run

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py
```