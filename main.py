import os
import pandas as pd
from preprocess import extract_features
from train import train_model, estimate_rul_all
from train_cnn_lstm import train_cnn_lstm


def create_combined_dataset():

    dataset_path = "battery_dataset"
    all_data = []

    for file in os.listdir(dataset_path):

        if file.endswith(".mat"):

            battery_name = file.split(".")[0]
            file_path = os.path.join(dataset_path, file)

            print(f"Processing {battery_name}...")

            cycle_data = extract_features(file_path, battery_name)

            for row in cycle_data:

                cycle, capacity, mean_voltage, mean_current, mean_temp, duration, soh = row

                all_data.append([
                    battery_name,
                    cycle,
                    capacity,
                    mean_voltage,
                    mean_current,
                    mean_temp,
                    duration,
                    soh
                ])

    df = pd.DataFrame(all_data, columns=[
        "Battery",
        "Cycle",
        "Capacity",
        "Mean_Voltage",
        "Mean_Current",
        "Mean_Temperature",
        "Discharge_Duration",
        "SoH"
    ])

    df.to_csv("combined_dataset.csv", index=False)

    print("\nFeature-Engineered Dataset Created!")
    print(df.head())
    print("\nTotal Data Points:", len(df))

    return "combined_dataset.csv"


def main():

    print("\n==============================")
    print(" EV Battery Health Prediction ")
    print("==============================\n")

    print("Step 1: Creating dataset...")
    csv_path = create_combined_dataset()

    print("\nStep 2: Training Random Forest Model...")
    rf_model, label_encoder = train_model(csv_path)

    print("\nStep 3: Training CNN + LSTM Deep Learning Model...")
    cnn_lstm_model = train_cnn_lstm(csv_path)

    print("\nStep 4: Estimating Remaining Useful Life (RUL)...")
    estimate_rul_all(csv_path)

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()