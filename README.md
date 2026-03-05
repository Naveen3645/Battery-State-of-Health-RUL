# 🔋 Battery State of Health (SoH) & Remaining Useful Life (RUL) Prediction

A **Hybrid Machine Learning + Deep Learning framework** for Lithium-ion battery health prediction using NASA battery aging datasets.

This project builds an **end-to-end predictive maintenance pipeline** capable of estimating battery degradation, predicting State of Health (SoH), and estimating Remaining Useful Life (RUL).

The system combines **Random Forest regression, CNN-LSTM deep learning, and degradation trend modeling**.

---

# 📌 Project Overview

Lithium-ion batteries gradually degrade as charge-discharge cycles increase. Monitoring this degradation is critical for:

- Electric Vehicles (EV)
- Battery Management Systems (BMS)
- Energy Storage Systems
- Aerospace applications

This project implements a **hybrid battery health prediction system** that:

- Extracts features from raw battery cycling data
- Calculates battery State of Health (SoH)
- Trains Machine Learning and Deep Learning models
- Predicts battery degradation behavior
- Estimates Remaining Useful Life (RUL)
- Visualizes battery degradation curves

---

# 🧠 Key Concepts

## 🔹 State of Health (SoH)

State of Health represents the remaining battery capacity relative to its rated capacity.

```
SoH (%) = (Current Capacity / Rated Capacity) × 100
```

Battery failure threshold:

```
70% SoH
```

---

## 🔹 Remaining Useful Life (RUL)

Remaining Useful Life represents how many cycles remain before the battery reaches failure.

```
RUL = Predicted Failure Cycle − Current Cycle
```

---

# 🏗 Project Architecture

```
NASA Battery Dataset (.mat)

        ↓

Feature Extraction
(preprocess.py)

        ↓

Feature Dataset
(combined_dataset.csv)

        ↓

Machine Learning Model
(Random Forest)

        ↓

Deep Learning Model
(CNN + LSTM Hybrid)

        ↓

Degradation Trend Modeling

        ↓

Remaining Useful Life (RUL)

        ↓

Battery Degradation Visualization
```

---

# 📂 Dataset

NASA Prognostics Center of Excellence (PCoE)  
Lithium-ion Battery Aging Dataset

Released by:

NASA Ames Research Center

Dataset Link:

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

### Dataset Characteristics

- 18650 Lithium-ion battery cells
- Charge / Discharge cycles
- Voltage measurements
- Current measurements
- Temperature measurements
- Capacity degradation over time

Example dataset files:

```
B0005.mat
B0006.mat
B0007.mat
B0018.mat
```

---

# 🛠 Features Extracted Per Cycle

From each discharge cycle:

- Mean Voltage
- Mean Current
- Mean Temperature
- Discharge Duration
- Capacity
- Computed SoH

These represent the electrochemical behavior of the battery.

---

# 🤖 Machine Learning Model

## Random Forest Regressor

Random Forest is used to predict battery State of Health.

Why Random Forest?

- Handles non-linear degradation
- Robust to noise
- Works well with tabular data
- Good generalization performance

Evaluation metrics:

- Mean Absolute Error (MAE)
- R² Score

---

# 🧠 Deep Learning Model

## CNN + LSTM Hybrid Model

To improve prediction accuracy, a **Convolutional Neural Network (CNN)** combined with **Long Short-Term Memory (LSTM)** is used.

### CNN Layer

Extracts signal patterns such as:

- Voltage fluctuations
- Current patterns
- Temperature behavior
- Discharge characteristics

### LSTM Layer

Captures temporal dependencies in battery degradation.

Example:

```
Cycle1 → Cycle2 → Cycle3 → Capacity drop
```

### Hybrid Architecture

```
Input Features

Cycle
Voltage
Current
Temperature

        ↓

CNN Layer
(Feature Extraction)

        ↓

LSTM Layer
(Time-Series Learning)

        ↓

Dense Layer

        ↓

Predicted SoH
```

---

# 📉 RUL Estimation Method

Remaining Useful Life is estimated using degradation trend modeling.

Steps:

1. Sort battery cycles
2. Apply Moving Average smoothing (window=10)
3. Fit Linear Regression to SoH vs Cycle
4. Estimate failure cycle at 70% SoH
5. Calculate Remaining Useful Life

This creates a hybrid **physics + data-driven model**.

---

# 📊 Visual Outputs

The project generates:

### SoH Prediction Plot

```
Actual SoH vs Predicted SoH
```

### Battery Degradation Curve

Graph containing:

- Raw SoH
- Smoothed SoH
- Linear degradation trend
- 70% failure threshold
- Predicted failure cycle

---

# 📁 Project Structure

```
Battery-State-of-Health-RUL
│
├── main.py
├── preprocess.py
├── train.py
├── train_cnn_lstm.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── battery_dataset
      ├── B0005.mat
      ├── B0006.mat
      ├── B0007.mat
      └── B0018.mat
```

---

# 📄 Output Files

```
combined_dataset.csv
rul_results.csv
cnn_lstm_battery_model.h5
SoH prediction plots
Battery degradation graphs
```

---

# 🚀 Applications

- Electric Vehicle Battery Monitoring
- Battery Management Systems (BMS)
- Renewable Energy Storage Systems
- Aerospace Battery Monitoring
- Predictive Maintenance systems

---

# 🎯 Skills Demonstrated

This project demonstrates:

- Battery degradation modeling
- Feature engineering
- Machine Learning (Random Forest)
- Deep Learning (CNN + LSTM)
- Time-series analysis
- Remaining Useful Life prediction
- Data visualization
- Predictive maintenance analytics

---

# 🔮 Future Improvements

Possible improvements:

- XGBoost model comparison
- Attention-based LSTM architecture
- Transformer-based battery prediction
- Cross-battery validation
- Real-time monitoring dashboard
- Web deployment using Flask / Streamlit
- Integration with Battery Management Systems
