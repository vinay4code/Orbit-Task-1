# Rocket Anomaly Detection Project
## Rocket Flight Anomaly Detection using Kalman Filter

This project simulates a two-stage rocket flight, generates synthetic sensor data, estimates flight states using a Kalman Filter, and detects anomalies in barometric readings based on residual analysis.

## 📁 Project Structure
```bash
Orbit Task 1/
├── README.md
├── report.pdf
├── src/
│   └── phase4_anomaly.py
|   |__ Anomaly_detection.py
├── data/
│   ├── sensor_data.csv
│   └── anomaly_report.csv
├── figures/
│   ├── altitude_estimation.png
│   └── residual.png

```
## 🚀 Features

Simulates realistic rocket flight dynamics

Adds noisy barometer and accelerometer sensor data

Applies Kalman Filter to estimate altitude and velocity

Computes residuals for barometer data

Detects anomalies using 3-sigma threshold

Logs anomalies and exports to CSV

Visualizes residuals and detected anomalies

## 🧪 Dependencies

Install with pip:
```bash
pip install numpy pandas matplotlib
```
## 📝 How to Run
```bash
python src/phase4_anomaly.py
python src/Anomaly_detection.py
```
Outputs:
```bash
data/sensor_data.csv

data/anomaly_report.csv

figures/residual.png
```
## 🤖 Model Details

Kalman Filter: Used for estimating altitude from accelerometer and barometer.

Anomaly Detection: No machine learning models are used; anomaly detection is purely statistical (residual > 3σ).

## 📊 Data Output

Simulated flight and sensor data are stored in data/simulated_flight_data.csv, including:

Time (s)

True Altitude (m)

True Velocity (m/s)

True Acceleration (m/s^2)

Barometer Reading (m)

Accelerometer Reading (m/s^2)

Anomaly log is stored in data/anomaly_report.csv.

## 📄 Report

See report.pdf for full project details, methodology, charts, and conclusion.
