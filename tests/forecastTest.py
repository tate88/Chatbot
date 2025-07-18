import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

data = {
    "date": ["2017-01-01", "2017-02-01", "2017-03-01", "2017-04-01", "2017-05-01", "2017-06-01", "2017-07-01", "2017-08-01", "2017-09-01", "2017-10-01",
             "2017-11-01", "2017-12-01", "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01", "2018-07-01", "2018-08-01",
             "2018-09-01", "2018-10-01", "2018-11-01", "2018-12-01", "2019-01-01", "2019-02-01", "2019-03-01", "2019-04-01", "2019-05-01", "2019-06-01",
             "2019-07-01", "2019-08-01", "2019-09-01", "2019-10-01", "2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
             "2020-05-01", "2020-06-01", "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01", "2020-11-01", "2020-12-01", "2021-01-01", "2021-02-01",
             "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-10-01", "2021-11-01", "2021-12-01",
             "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01",
             "2022-11-01", "2022-12-01", "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
    "sales_amount": [30104.00, 11393.00, 16071.50, 21631.50, 24970.32, 17351.50, 53068.00, 4519.50, 9137.00, 17703.50,
                     15846.00, 9108.00, 13926.50, 12981.50, 6136.00, 21445.20, 14857.50, 12584.50, 6285.50, 7853.00,
                     12203.30, 5282.10, 5363.50, 5359.85, 3294.00, 6758.50, 10218.50, 3863.50, 8739.70, 4893.00,
                     2517.00, 15302.80, 3611.50, 5152.00, 4134.50, 8622.00, 7192.75, 9241.50, 29125.50, 22519.10,
                     5805.00, 835.84, 5911.00, 0.00, 9233.10, 17403.50, 9310.50, 7306.35, 9158.65, 4317.90,
                     5291.50, 8307.50, 38283.50, 8725.45, 25124.26, 13389.50, 18770.10, 4489.50, 5385.50, 896.00,
                     1792.00, 896.00, 896.00, 896.00, 896.00, 2048.00, 896.00, 1647.00, 896.00, 2422.00,
                     1792.00, 3515.00, 7498.00, 6620.00, 774.00, 1677.00, 2881.00]
}
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ==== 2. Preprocessing ====
df['sales_amount'] = df['sales_amount'].clip(lower=1e-3)  # Prevent log(0)
df['sales_log'] = np.log1p(df['sales_amount'])  # log transform

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['sales_log']])

# ==== 3. Create Sequences ====
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 3
X, y = create_sequences(scaled_data, window_size)

# ==== 4. Train/Test Split ====
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==== 5. Build & Train LSTM Model ====
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=300, verbose=0)

# ==== 6. Predict ====
predictions = model.predict(X_test)
y_test_inv = np.expm1(scaler.inverse_transform(y_test.reshape(-1, 1)))
predictions_inv = np.expm1(scaler.inverse_transform(predictions))

# ==== 7. Accuracy ====
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
print(f"✅ LSTM Forecast RMSE: {rmse:.2f}")

# ==== 8. Plot ====
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(predictions_inv, label='LSTM Forecast')
plt.title("📈 LSTM Forecast vs Actual")
plt.xlabel("Time Index")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
