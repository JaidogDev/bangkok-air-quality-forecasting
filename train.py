import os
import json
import random
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# from model import build_lstm_model
from model_improve import build_lstm_model


CSV_PATH = "data/bangkok-air-quality.csv"
WINDOW_SIZE = 7
EPOCHS = 50
BATCH_SIZE = 32
SEED = 42

FEATURE_COLUMNS = ["pm25", "pm10"]
TARGET_COLUMN = "pm25"

EXPERIMENT_NAME = "improved_lstm_dense_tanh_dropout50"
MODEL_DIR = os.path.join("saved_models", EXPERIMENT_NAME)
OUTPUT_DIR = os.path.join("outputs", EXPERIMENT_NAME)

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def create_sequences(data, window_size):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])  # target = pm25

    return np.array(X), np.array(y)


def inverse_pm25(scaler, values, n_features):
    """
    inverse transform เฉพาะ pm25
    เพราะ scaler ถูก fit ด้วย feature 2 ตัว: pm25, pm10
    """
    dummy = np.zeros((len(values), n_features))
    dummy[:, 0] = values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]


def main():
    # os.makedirs("saved_models", exist_ok=True)
    # os.makedirs("outputs", exist_ok=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # 1) Load and clean dataset
    # =========================
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    df = df[["date", "pm25", "pm10"]].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")

    original_rows = len(df)

    df = df.dropna(subset=["pm25", "pm10"]).reset_index(drop=True)

    cleaned_rows = len(df)
    dropped_rows = original_rows - cleaned_rows

    print("Original rows:", original_rows)
    print("After drop missing:", df.shape)
    print("Dropped rows:", dropped_rows)
    print("Start date:", df["date"].min())
    print("End date:", df["date"].max())

    print("Date gap count:")
    date_gap_count = df["date"].diff().value_counts().head()
    print(date_gap_count)

    print("Missing values:")
    print(df.isna().sum())

    # =========================
    # 2) Prepare features
    # =========================
    features = df[FEATURE_COLUMNS].values

    split_index = int(len(features) * 0.8)

    train_features = features[:split_index]

    test_features = features[split_index - WINDOW_SIZE:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # =========================
    # 3) Build model
    # =========================
    model = build_lstm_model(
        window_size=WINDOW_SIZE,
        n_features=X_train.shape[2]
    )

    model.summary()

    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # =========================
    # 4) Callbacks
    # =========================
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_lstm_pm25_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        CSVLogger(os.path.join(OUTPUT_DIR, "training_history.csv"))
    ]

    # =========================
    # 5) Train model
    # =========================
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "final_lstm_pm25_model.keras"))

    # =========================
    # 6) Load best model for test evaluation
    # =========================
    best_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "best_lstm_pm25_model.keras")
    )

    y_pred = best_model.predict(X_test)

    n_features = len(FEATURE_COLUMNS)

    y_pred_real = inverse_pm25(
        scaler=scaler,
        values=y_pred,
        n_features=n_features
    )

    y_test_real = inverse_pm25(
        scaler=scaler,
        values=y_test,
        n_features=n_features
    )

    # =========================
    # 7) LSTM Metrics
    # =========================
    lstm_mae = mean_absolute_error(y_test_real, y_pred_real)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print("Best LSTM MAE:", lstm_mae)
    print("Best LSTM RMSE:", lstm_rmse)

    # =========================
    # 8) Naive baseline
    # =========================
    naive_scaled = X_test[:, -1, 0]

    naive_pred_real = inverse_pm25(
        scaler=scaler,
        values=naive_scaled,
        n_features=n_features
    )

    naive_mae = mean_absolute_error(y_test_real, naive_pred_real)
    naive_rmse = np.sqrt(mean_squared_error(y_test_real, naive_pred_real))

    print("Naive MAE:", naive_mae)
    print("Naive RMSE:", naive_rmse)

    if lstm_mae < naive_mae:
        result_text = "LSTM is better than naive baseline based on MAE."
    else:
        result_text = "LSTM is not better than naive baseline based on MAE."

    print("Result:", result_text)

    # =========================
    # 9) Best epoch details
    # =========================
    val_losses = history.history["val_loss"]
    val_maes = history.history["val_mae"]

    best_epoch_index = int(np.argmin(val_losses))
    best_epoch = best_epoch_index + 1
    best_val_loss = float(val_losses[best_epoch_index])
    best_val_mae = float(val_maes[best_epoch_index])

    print("Best epoch:", best_epoch)
    print("Best val_loss:", best_val_loss)
    print("Best val_mae:", best_val_mae)

    # =========================
    # 10) Save scaler
    # =========================
    joblib.dump(scaler, os.path.join(MODEL_DIR, "pm25_scaler.pkl"))

    print(f"Best model saved to: {os.path.join(MODEL_DIR, 'best_lstm_pm25_model.keras')}")
    print(f"Final model saved to: {os.path.join(MODEL_DIR, 'final_lstm_pm25_model.keras')}")
    print(f"Scaler saved to: {os.path.join(MODEL_DIR, 'pm25_scaler.pkl')}")

    # =========================
    # 11) Save metrics/details JSON
    # =========================
    details = {
        "project": "PM2.5 Forecasting using LSTM",
        "dataset_path": CSV_PATH,
        "target": "next_day_pm25",
        "features": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "original_rows": int(original_rows),
        "cleaned_rows": int(cleaned_rows),
        "dropped_rows": int(dropped_rows),
        "start_date": str(df["date"].min()),
        "end_date": str(df["date"].max()),
        "train_feature_rows": int(len(train_features)),
        "test_feature_rows_with_context": int(len(test_features)),
        "x_train_shape": list(X_train.shape),
        "x_test_shape": list(X_test.shape),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "test_lstm_mae": float(lstm_mae),
        "test_lstm_rmse": float(lstm_rmse),
        "test_naive_mae": float(naive_mae),
        "test_naive_rmse": float(naive_rmse),
        "result": result_text,
        "experiment_name": EXPERIMENT_NAME,
        "best_model_path": os.path.join(MODEL_DIR, "best_lstm_pm25_model.keras"),
        "final_model_path": os.path.join(MODEL_DIR, "final_lstm_pm25_model.keras"),
        "scaler_path": os.path.join(MODEL_DIR, "pm25_scaler.pkl"),
        "training_history_path": os.path.join(OUTPUT_DIR, "training_history.csv"),
        "prediction_csv_path": os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        "prediction_plot_path": os.path.join(OUTPUT_DIR, "pm25_prediction_plot.png"),
        "loss_plot_path": os.path.join(OUTPUT_DIR, "loss_plot.png"),
        "model_summary_path": os.path.join(OUTPUT_DIR, "model_summary.txt")
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=4, ensure_ascii=False)

    print(f"Metrics/details saved to: {metrics_path}")

    # =========================
    # 12) Save predictions CSV
    # =========================
    test_dates = df["date"].iloc[split_index:].reset_index(drop=True)

    prediction_df = pd.DataFrame({
        "date": test_dates,
        "actual_pm25": y_test_real,
        "lstm_predicted_pm25": y_pred_real,
        "naive_predicted_pm25": naive_pred_real,
        "lstm_error": y_pred_real - y_test_real,
        "naive_error": naive_pred_real - y_test_real,
        "absolute_lstm_error": np.abs(y_pred_real - y_test_real),
        "absolute_naive_error": np.abs(naive_pred_real - y_test_real)
    })

    print(prediction_df.head(20))

    prediction_csv_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")

    prediction_df.to_csv(prediction_csv_path, index=False)

    print(f"Predictions saved to: {prediction_csv_path}")

    # =========================
    # 13) Plot actual vs predicted
    # =========================
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_real, label="Actual PM2.5")
    plt.plot(y_pred_real, label="Best LSTM Predicted PM2.5")
    plt.plot(naive_pred_real, label="Naive Baseline", linestyle="--")
    plt.legend()
    plt.title("PM2.5 Prediction: Best LSTM vs Naive Baseline")
    plt.xlabel("Test Time Step")
    plt.ylabel("PM2.5")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pm25_prediction_plot.png"), dpi=300)
    plt.show()

    # =========================
    # 14) Plot training loss
    # =========================
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.axvline(
        x=best_epoch_index,
        linestyle="--",
        label=f"Best Epoch: {best_epoch}"
    )
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()