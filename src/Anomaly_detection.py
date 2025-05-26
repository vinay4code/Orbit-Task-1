import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ======= Rocket Simulation =======
def simulate_rocket(duration=120, dt=0.1):
    time = np.arange(0, duration, dt)
    altitude = []
    acceleration = []

    for t in time:
        if t < 2:
            a = 0
        elif t < 20:
            a = 40 - 9.81
        elif t < 21:
            a = -15
        elif t < 50:
            a = 20 - 9.81
        else:
            a = -9.81
        acceleration.append(a)

    acceleration = np.array(acceleration)
    velocity = np.cumsum(acceleration) * dt
    altitude = np.cumsum(velocity) * dt
    return time, altitude, velocity, acceleration

# ======= Sensor Simulation =======
def add_noise(data, std_dev=0.05, bias=0.0):
    noise = np.random.normal(0, std_dev, size=data.shape)
    return data + noise + bias

def simulate_barometer(altitude):
    P0 = 101325
    T0 = 288.15
    L = 0.0065
    R = 287.05
    g = 9.81
    T = np.clip(T0 - L * altitude, 1.0, None)
    P = P0 * (T / T0) ** (g / (R * L))
    h = 44330 * (1 - (P / P0) ** (1/5.255))
    return add_noise(h, std_dev=5)

def simulate_accelerometer(true_acc):
    return add_noise(true_acc, std_dev=0.3)

def simulate_sensors(time, altitude, acceleration):
    return {
        "barometer": simulate_barometer(altitude),
        "accelerometer": simulate_accelerometer(acceleration)
    }

# ======= Kalman Filter + Residuals =======
def kalman_filter_with_residuals(time, accel_data, baro_data, dt):
    n = len(time)
    x = np.zeros((2, n))  # [altitude, velocity]
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5 * dt**2], [dt]])
    H = np.array([[1, 0]])

    P = np.eye(2)
    Q = np.array([[1, 0], [0, 3]]) * 0.05
    R = np.array([[20**2]])

    est_altitude = []
    residuals = []

    for i in range(1, n):
        u = accel_data[i]
        z = baro_data[i]

        # Predict
        x_pred = A @ x[:, i-1] + B.flatten() * u
        P = A @ P @ A.T + Q

        # Update
        y = z - H @ x_pred  # residual
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x[:, i] = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P

        est_altitude.append(x[0, i])
        residuals.append(y.item())

    return np.array(est_altitude), np.array(residuals)

# ======= Anomaly Detection =======
def detect_anomalies(residuals, threshold_sigma=3):
    residuals = np.array(residuals)
    mean = np.mean(residuals)
    std = np.std(residuals)
    anomalies = np.abs(residuals - mean) > threshold_sigma * std
    return anomalies

def log_anomalies(time, baro_data, residuals, anomalies):
    log = []
    mean = np.mean(residuals)
    std = np.std(residuals)
    
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            residual = residuals[i]
            severity = np.abs(residual - mean) / std
            log.append({
                "Time (s)": time[i+1],
                "Barometer Altitude (m)": baro_data[i+1],
                "Residual": residual,
                "Severity (sigma)": round(severity, 2)
            })
    return pd.DataFrame(log)


# ======= Main =======
# Use this after running detect_anomalies()
if __name__ == "__main__":
    dt = 0.1
    time, alt, vel, acc = simulate_rocket(dt=dt)
    sensor_data = simulate_sensors(time, alt, acc)

    est_alt, residuals = kalman_filter_with_residuals(
        time, sensor_data["accelerometer"], sensor_data["barometer"], dt
    )
    anomalies = detect_anomalies(residuals)

    # ---- Phase 6: Post-Flight Logging ----
    anomaly_log = log_anomalies(time, sensor_data["barometer"], residuals, anomalies)
    anomaly_log.to_csv("anomaly_report.csv", index=False)
    print("=== Anomaly Log ===")
    print(anomaly_log)

    # ---- Plot residuals with anomalies ----
    plt.figure(figsize=(12, 5))
    plt.plot(time[1:], residuals, label="Residuals")
    plt.scatter(time[1:][anomalies], residuals[anomalies], color='red', label="Anomalies", marker='x')
    plt.axhline(y=3*np.std(residuals), color='gray', linestyle='--', label="Threshold")
    plt.axhline(y=-3*np.std(residuals), color='gray', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Residual")
    plt.title("Residuals with Detected Anomalies")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()