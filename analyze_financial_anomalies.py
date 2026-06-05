import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import signal


def signal_handler(sig, frame):
    print("\n\n⚠️  Process interrupted by user. Exiting gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def load_data(path):
    print(" Loading data...", end=" ", flush=True)
    df = pd.read_csv(path, parse_dates=["Timestamp"], dayfirst=True)
    print(f"✓ Loaded {len(df)} records")
    return df


def preprocess(df):
    print("⚙️  preprocessing data...", end=" ", flush=True)
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.day_name()
    df["IsHighAmount"] = df["Amount"] > df["Amount"].quantile(0.95)
    print("✓ Complete")
    return df


def summary(df):
    print("\n📊 DATA SUMMARY")
    print("=" * 50)
    print(df.describe(include="all"))
    print("\n🔄 Transaction types:")
    print(df["TransactionType"].value_counts())
    print("\n📍 Top locations:")
    print(df["Location"].value_counts().head(10))
    print("\n🏪 Top merchants:")
    print(df["Merchant"].value_counts().head(10))


def detect_anomalies(df):
    print("\n🔍 Detecting anomalies...", flush=True)
    print("  → Feature encoding...", end=" ", flush=True)
    features = pd.get_dummies(df[["Amount", "Hour", "TransactionType", "Location", "Merchant"]], drop_first=True)
    print(f"✓ ({features.shape[1]} features)")
    
    print("  → Training Isolation Forest...", end=" ", flush=True)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(features)
    print("✓")
    
    print("  → Scoring transactions...", end=" ", flush=True)
    df["AnomalyScore"] = model.decision_function(features)
    predictions = model.predict(features)
    df["Anomaly"] = (predictions == -1).astype(int)
    print("✓")
    return df


def report_anomalies(df, path):
    print("📋 Generating anomaly report...", end=" ", flush=True)
    anomalies = df[df["Anomaly"] == 1].sort_values("AnomalyScore")
    anomalies.to_csv(path, index=False)
    print(f"✓ Saved {len(anomalies)} anomalies to {path}")
    return anomalies


def plot_analysis(df):
    print("📈 Creating visualizations...", flush=True)
    
    print("  → Plotting transaction amount distribution...", end=" ", flush=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Amount"], bins=30, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.tight_layout()
    plt.savefig("transaction_amount_distribution.png", dpi=100)
    plt.close()
    print("✓")

    print("  → Plotting amount by transaction type...", end=" ", flush=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="TransactionType", y="Amount", data=df)
    plt.title("Amount by Transaction Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("amount_by_transaction_type.png", dpi=100)
    plt.close()
    print("✓")

    print("  → Plotting transactions by day of week...", end=" ", flush=True)
    plt.figure(figsize=(10, 6))
    day_counts = df["DayOfWeek"].value_counts()
    sns.barplot(x=day_counts.index, y=day_counts.values)
    plt.title("Transactions by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig("transactions_by_day_of_week.png", dpi=100)
    plt.close()
    print("✓")
    print("✅ All plots saved successfully")


def main():
    try:
        print("\n" + "=" * 50)
        print("💰 FINANCIAL ANOMALY DETECTION SYSTEM💰")
        print("=" * 50 + "\n")
        
        csv_path = "financial_anomaly_data.csv"
        df = load_data(csv_path)
        df = preprocess(df)
        summary(df)
        df = detect_anomalies(df)
        anomalies = report_anomalies(df, "anomaly_report.csv")
        
        print(f"\n⚠️  Found {len(anomalies)} anomaly transactions ({len(anomalies)/len(df)*100:.2f}%)")
        if len(anomalies) > 0:
            print("\n🔴 Top 20 Most Anomalous Transactions:")
            print("-" * 50)
            print(anomalies[["Timestamp", "TransactionID", "AccountID", "Amount", "Merchant", "TransactionType", "Location", "AnomalyScore"]].head(20).to_string(index=False))
        
        plot_analysis(df)
        
        print("\n" + "=" * 50)
        print("✅ Analysis complete!")
        print("=" * 50 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
