import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def train_cnn_lstm(csv_path):

    df = pd.read_csv(csv_path)

    # Encode battery names
    le = LabelEncoder()
    df["Battery_encoded"] = le.fit_transform(df["Battery"])

    features = df[[
        "Cycle",
        "Mean_Voltage",
        "Mean_Current",
        "Mean_Temperature",
        "Discharge_Duration",
        "Battery_encoded"
    ]]

    target = df["SoH"]

    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Create time sequences
    time_steps = 10

    X = []
    y = []

    for i in range(len(features_scaled) - time_steps):
        X.append(features_scaled[i:i+time_steps])
        y.append(target.iloc[i+time_steps])

    X = np.array(X)
    y = np.array(y)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Data Shape:", X_train.shape)

    # CNN + LSTM Hybrid Model
    model = Sequential()

    model.add(Conv1D(
        filters=64,
        kernel_size=2,
        activation='relu',
        input_shape=(time_steps, X.shape[2])
    ))

    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(64, return_sequences=True))

    model.add(LSTM(32))

    model.add(Dropout(0.3))

    model.add(Dense(16, activation="relu"))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("\nTraining CNN + LSTM Model...\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    model.save("cnn_lstm_battery_model.h5")

    print("\nCNN + LSTM Model Trained Successfully!")

    # Plot loss graph
    plt.figure(figsize=(7,5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN + LSTM Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model