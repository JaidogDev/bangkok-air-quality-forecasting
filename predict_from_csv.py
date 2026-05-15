import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


MODEL_PATH = "saved_models/best_lstm_pm25_model.keras"
SCALER_PATH = "saved_models/pm25_scaler.pkl"
INPUT_PATH = "data/latest_7_input.csv"

WINDOW_SIZE = 7
FEATURE_COLUMNS = ["pm25", "pm10"]


def inverse_pm25(scaler, value, n_features=2):
    dummy = np.zeros((1, n_features))
    dummy[0, 0] = value
    return scaler.inverse_transform(dummy)[0, 0]


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(INPUT_PATH)
    df.columns = df.columns.str.strip()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")

    df = df.dropna(subset=["pm25", "pm10"]).reset_index(drop=True)

    if len(df) != WINDOW_SIZE:
        raise ValueError(f"Input CSV must contain exactly {WINDOW_SIZE} valid rows, but got {len(df)} rows.")

    latest_7 = df[FEATURE_COLUMNS].values

    latest_7_scaled = scaler.transform(latest_7)
    X_input = latest_7_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS))

    pred_scaled = model.predict(X_input, verbose=0)

    pred_pm25 = inverse_pm25(
        scaler=scaler,
        value=pred_scaled[0, 0],
        n_features=len(FEATURE_COLUMNS)
    )

    latest_date = df["date"].iloc[-1]
    predicted_date = latest_date + pd.Timedelta(days=1)

    print("Input data:")
    print(df[["date", "pm25", "pm10"]])

    print("\nPrediction result")
    print("Latest input date:", latest_date.date())
    print("Predicted date:", predicted_date.date())
    print(f"Predicted PM2.5: {pred_pm25:.2f} µg/m³")


if __name__ == "__main__":
    main()