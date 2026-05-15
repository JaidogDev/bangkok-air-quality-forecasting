from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


def build_lstm_model(window_size: int, n_features: int):
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(32),
        Dropout(0.2),
        # Dense(16, activation="relu"),
        Dense(16, activation="tanh"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model