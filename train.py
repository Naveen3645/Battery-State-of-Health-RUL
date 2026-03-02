import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression


def train_model(csv_path):

    df = pd.read_csv(csv_path)

    le = LabelEncoder()
    df["Battery_encoded"] = le.fit_transform(df["Battery"])

    X = df[[
        "Cycle",
        "Mean_Voltage",
        "Mean_Current",
        "Mean_Temperature",
        "Discharge_Duration",
        "Battery_encoded"
    ]]

    y = df["SoH"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nRandom Forest Model Trained Successfully!")
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual SoH")
    plt.ylabel("Predicted SoH")
    plt.title("Actual vs Predicted SoH")
    plt.grid(True)
    plt.show()

    return model, le


def estimate_rul_all(csv_path):

    df = pd.read_csv(csv_path)

    print("\n===== HYBRID RUL ESTIMATION (Smoothed Degradation Model) =====")

    results = []

    for battery_name in df["Battery"].unique():

        battery_df = df[df["Battery"] == battery_name].copy()
        battery_df = battery_df.sort_values("Cycle")

        battery_df["SoH_Smoothed"] = battery_df["SoH"].rolling(
            window=10,
            min_periods=1
        ).mean()

        X_trend = battery_df[["Cycle"]]
        y_trend = battery_df["SoH_Smoothed"]

        trend_model = LinearRegression()
        trend_model.fit(X_trend, y_trend)

        slope = trend_model.coef_[0]
        intercept = trend_model.intercept_

        current_cycle = battery_df["Cycle"].max()

        if slope >= 0:
            print(f"{battery_name} → Invalid degradation trend (after smoothing)")
            failure_cycle = current_cycle
            rul = 0
        else:
            failure_cycle = (70 - intercept) / slope
            rul = failure_cycle - current_cycle

            if failure_cycle < 0:
                failure_cycle = current_cycle
                rul = 0

            if rul < 0:
                rul = 0

        print(f"{battery_name} → Current: {current_cycle:.0f} | "
              f"Predicted Failure: {failure_cycle:.0f} | "
              f"RUL: {rul:.0f} cycles")

        # 🔹 Final Degradation Graph
        plt.figure(figsize=(8, 6))

        plt.plot(
            battery_df["Cycle"],
            battery_df["SoH"],
            alpha=0.4,
            label="Raw SoH"
        )

        plt.plot(
            battery_df["Cycle"],
            battery_df["SoH_Smoothed"],
            linewidth=2,
            label="Smoothed SoH"
        )

        predicted_trend = trend_model.predict(X_trend)

        plt.plot(
            battery_df["Cycle"],
            predicted_trend,
            linestyle="--",
            label="Degradation Trend"
        )

        plt.axhline(
            y=70,
            linestyle=":",
            linewidth=2,
            label="Failure Threshold (70%)"
        )

        if slope < 0 and failure_cycle > current_cycle:
            plt.axvline(
                x=failure_cycle,
                linestyle=":",
                linewidth=2,
                label="Predicted Failure Cycle"
            )

        plt.xlabel("Cycle")
        plt.ylabel("State of Health (%)")
        plt.title(f"Battery Degradation Curve - {battery_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        results.append([
            battery_name,
            current_cycle,
            failure_cycle,
            rul
        ])

    rul_df = pd.DataFrame(results, columns=[
        "Battery",
        "Current_Cycle",
        "Failure_Cycle",
        "RUL"
    ])

    rul_df.to_csv("rul_results.csv", index=False)

    print("\nSmoothed RUL results saved to rul_results.csv")