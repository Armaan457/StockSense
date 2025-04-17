import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Config
ticker = "AAPL"
start_date = "2023-01-01"
end_date = "2025-04-17"
contamination = 0.05
rolling_window = 20

# Fetch stock data
df = yf.download(ticker, start=start_date, end=end_date)

# Handle missing values
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# Select features and scale them
features = df[["Open", "High", "Low", "Close", "Volume"]]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Apply Isolation Forest on price features
model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
df["Anomaly"] = model.fit_predict(scaled_features)

# Rolling metrics for context
df["Rolling_Mean"] = df["Close"].rolling(window=rolling_window).mean()
df["Rolling_Std"] = df["Close"].rolling(window=rolling_window).std()

# Plot closing price with anomalies and rolling stats
anomalies = df[df["Anomaly"] == -1]
plt.figure(figsize=(14, 7))
plt.plot(df.index, df["Close"], label="Closing Price", color="blue")
plt.plot(df.index, df["Rolling_Mean"], label=f"{rolling_window}-Day MA", color="orange", linestyle="--")
plt.fill_between(df.index,
                 df["Rolling_Mean"] - 2 * df["Rolling_Std"],
                 df["Rolling_Mean"] + 2 * df["Rolling_Std"],
                 color="orange", alpha=0.2, label="±2 Std Dev")
plt.scatter(anomalies.index, anomalies["Close"], color="red", label="Anomalies", marker="o")

# Add annotations for anomalies
for i in anomalies.index:
    plt.annotate(i.strftime('%Y-%m-%d'), (i, anomalies.loc[i, "Close"]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='darkred')

plt.title(f"{ticker} Price Anomaly Detection with Rolling Stats")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Feature importance (if supported)
if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": features.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    print("Feature importances:\n", importance_df)

    importance_df.plot(kind="bar", x="Feature", y="Importance", legend=False,
                       title="Feature Importance (Isolation Forest)", figsize=(8, 4), color="skyblue")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()

# Return-based anomaly detection
df["Returns"] = df["Close"].pct_change().fillna(0)
return_scaled = scaler.fit_transform(df[["Returns"]])
return_model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
df["Return_Anomaly"] = return_model.fit_predict(return_scaled)

# Plot anomalies in returns
return_anomalies = df[df["Return_Anomaly"] == -1]
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["Returns"], label="Daily Returns", color="purple")
plt.scatter(return_anomalies.index, return_anomalies["Returns"], color="red", label="Anomalies", marker="x")
plt.title(f"{ticker} Return-Based Anomaly Detection")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Save CSV
df.to_csv(f"{ticker}_anomalies.csv")
print(f"CSV saved as {ticker}_anomalies.csv")

# Optional: Excel with highlighted anomalies
try:
    with pd.ExcelWriter(f"{ticker}_anomaly_report.xlsx", engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Anomalies")
        workbook = writer.book
        worksheet = writer.sheets["Anomalies"]
        red_format = workbook.add_format({'bg_color': '#FFC7CE'})

        for row_num, val in enumerate(df["Anomaly"], start=1):
            if val == -1:
                worksheet.set_row(row_num, cell_format=red_format)
    print(f"Excel report saved as {ticker}_anomaly_report.xlsx")
except Exception as e:
    print(f"Excel export failed: {e}")
