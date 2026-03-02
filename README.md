# Battery-State-of-Health-RUL
Lithium-ion Battery State of Health (SoH) and Remaining Useful Life (RUL) prediction framework using NASA battery aging datasets with ML-based degradation modeling.
# 🔋 Battery State of Health (SoH) & Remaining Useful Life (RUL) Prediction

An end-to-end Predictive Maintenance framework for Lithium-ion batteries using machine learning and degradation trend modeling based on NASA battery aging datasets.

---

## 📌 Project Overview

Lithium-ion batteries degrade over charge-discharge cycles, leading to reduced capacity and eventual failure.  
This project implements a complete battery prognostics pipeline to:

- Extract features from raw battery cycling data
- Estimate State of Health (SoH)
- Train a Machine Learning regression model
- Predict Remaining Useful Life (RUL)
- Visualize battery degradation trends

This system simulates a simplified Battery Management System (BMS) health prediction module.

---

## 🧠 Key Concepts

### 🔹 State of Health (SoH)

SoH represents the current capacity of the battery relative to its rated capacity:

SoH (%) = (Current Capacity / Rated Capacity) × 100

Failure threshold assumed at 70% SoH.

---

### 🔹 Remaining Useful Life (RUL)

RUL is the estimated number of cycles remaining before the battery reaches the failure threshold.

RUL = Predicted Failure Cycle − Current Cycle

---

## 🏗 Project Architecture

Raw .mat Data  
→ Feature Extraction  
→ SoH Calculation  
→ Machine Learning Model  
→ Degradation Trend Modeling  
→ RUL Estimation  
→ Visualization  

---

## 📂 Dataset

NASA Prognostics Center of Excellence (PCoE) Lithium-ion Battery Dataset

Released by:
NASA Ames Research Center

Download Link:
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

### Dataset Characteristics

- 18650 Lithium-ion cells
- Charge/Discharge cycles
- Measured Voltage, Current, Temperature
- Capacity degradation over time
- End-of-life around 70% capacity

Example files:
- B0005.mat
- B0006.mat
- B0007.mat
- B0018.mat

---

## 🛠 Features Extracted Per Cycle

From each discharge cycle:

- Mean Voltage
- Mean Current
- Mean Temperature
- Discharge Duration
- Capacity
- Computed SoH

---

## 🤖 Machine Learning Model

Model Used:
Random Forest Regressor

Why Random Forest?
- Handles non-linearity
- Robust to noise
- Good generalization
- Works well with tabular battery features

Evaluation Metrics:
- Mean Absolute Error (MAE)
- R² Score

---

## 📉 RUL Estimation Method

1. Sort battery cycles
2. Apply Moving Average smoothing (window=10)
3. Fit Linear Regression to SoH vs Cycle
4. Compute failure cycle at 70% SoH
5. Calculate Remaining Useful Life

This creates a hybrid physics + data-driven degradation model.

---

## 📊 Visual Outputs

- Actual vs Predicted SoH scatter plot
- Final degradation curve showing:
  - Raw SoH
  - Smoothed SoH
  - Linear degradation trend
  - 70% failure threshold
  - Predicted failure cycle

---

## 📁 Project Structure


Battery-State-of-Health-RUL/
│
├── main.py             
├── preprocess.py      
├── train.py             
├── requirements.txt   
├── README.md            
├── .gitignore           
│
└── battery_dataset/     


---

## 📄 Output Files

- combined_dataset.csv
- rul_results.csv
- SoH prediction plot
- Degradation curve plot

---

## 🚀 Applications

- Electric Vehicles (EV)
- Battery Management Systems (BMS)
- Energy Storage Systems
- Aerospace battery health monitoring
- Predictive Maintenance systems

---

## 🎯 Skills Demonstrated

- Battery degradation modeling
- Signal-based feature engineering
- Machine Learning regression
- Random Forest implementation
- Trend analysis & smoothing
- Failure threshold modeling
- Data visualization
- Predictive maintenance logic

---

## 🔮 Future Improvements

- Add XGBoost model comparison
- Add LSTM-based time series modeling
- Use exponential degradation fitting
- Add cross-battery validation
- Deploy as web dashboard
- Save trained model as .pkl

---
