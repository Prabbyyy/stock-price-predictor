# ============================================================
# PROJECT 1: Stock Price Trend Predictor
# Tools: Python, yfinance, scikit-learn, Pandas, Matplotlib
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Fetch Data ─────────────────────────────────────────
ticker = "RELIANCE.NS"          # Change to any NSE ticker
df = yf.download(ticker, start="2022-01-01", end="2024-12-31", auto_adjust=True)
# Flatten MultiIndex columns if present (yfinance v0.2+)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[['Close']].copy()
df.dropna(inplace=True)
print(f"Downloaded {len(df)} rows for {ticker}")

# ── 2. Feature Engineering ────────────────────────────────
df['MA20']  = df['Close'].rolling(20).mean()          # 20-day moving average
df['MA50']  = df['Close'].rolling(50).mean()          # 50-day moving average

# RSI (Relative Strength Index)
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Lag features
df['Lag1'] = df['Close'].shift(1)
df['Lag3'] = df['Close'].shift(3)
df['Lag5'] = df['Close'].shift(5)

# Target: next day close
df['Target'] = df['Close'].shift(-1)

df.dropna(inplace=True)

# ── 3. Train / Test Split ─────────────────────────────────
features = ['MA20', 'MA50', 'RSI', 'Lag1', 'Lag3', 'Lag5']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False   # time-series: no shuffle
)

# ── 4. Model Training ─────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 5. Evaluation ─────────────────────────────────────────
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"  Mean Absolute Error : ₹{mae:.2f}")
print(f"  R² Score            : {r2:.4f}")

# ── 6. Directional Accuracy ───────────────────────────────
actual_dir    = np.sign(y_test.values[1:] - y_test.values[:-1])
predicted_dir = np.sign(y_pred[1:] - y_pred[:-1])
dir_accuracy  = np.mean(actual_dir == predicted_dir) * 100
print(f"  Directional Accuracy: {dir_accuracy:.2f}%")

# ── 7. Plot ───────────────────────────────────────────────
plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test.values, label="Actual Price", color="steelblue", linewidth=1.5)
plt.plot(y_test.index, y_pred,        label="Predicted Price", color="orange",
         linewidth=1.5, linestyle="--")
plt.title(f"{ticker} — Actual vs Predicted Close Price (Random Forest)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.tight_layout()
plt.savefig("stock_prediction.png", dpi=150)
plt.show()
print("Plot saved as stock_prediction.png")
