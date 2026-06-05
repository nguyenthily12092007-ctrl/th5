# 💰 Financial Anomaly Detection System

## Overview
Web-based anomaly detection system built with **Streamlit** for analyzing financial transactions using **Isolation Forest** machine learning algorithm.

## 📦 Files Created

### Main Application
- **app.py** - Streamlit web application (purple-themed interface)
- **analyze_financial_anomalies.py** - Command-line analysis script
- **environment.yml** - Conda environment configuration
- **financial_anomaly_data.csv** - Transaction dataset (217,441 records)

### Generated Reports
- **anomaly_report.csv** - Detected anomalies with scores
- **transaction_amount_distribution.png** - Amount distribution chart
- **amount_by_transaction_type.png** - Amount by transaction type chart
- **transactions_by_day_of_week.png** - Daily transaction pattern chart

## 🚀 Getting Started

### Option 1: Run Streamlit Web App (Recommended)
```powershell
cd c:\Users\pc\Downloads\Th5
conda activate financial-anomaly
streamlit run app.py
```
Or run the bundled launcher:
```powershell
cd c:\Users\pc\Downloads\Th5
conda activate financial-anomaly
python run_app.py
```
Then open: **http://localhost:8501**

You can also upload your own transaction CSV directly in the app sidebar.

### Option 2: Run Command-Line Analysis
```powershell
cd c:\Users\pc\Downloads\Th5
conda activate financial-anomaly
python analyze_financial_anomalies.py
```

## 📊 Application Features

### Key Metrics Dashboard
- **📊 Total Transactions**: Count of all transactions analyzed
- **⚠️ Anomalies Detected**: Number of suspicious transactions found
- **📊 Risk Rate**: Percentage of anomalous transactions
- **💸 Suspicious Amount**: Total monetary value of anomalies

### Four Analysis Tabs
1. **📊 Overview** - Transaction distribution by type and location
2. **🔴 Anomalies** - Detailed anomalous transaction analysis
3. **📈 Analytics** - Statistical visualizations and distributions
4. **🔍 Detailed View** - Filterable transaction table with download

### Sidebar Controls
- **Number of Estimators**: 50-300 (accuracy vs speed trade-off)
- **Contamination Rate**: 1-20% (expected anomaly percentage)
- **Date Range Filter**: Custom time period selection

## 🎨 Design Features
- **Color Theme**: Purple (#6a1b9a primary, #e1bee7 cards)
- **Background**: Light purple (#f3e5f5)
- **Risk Indicators**: Red for high risk, orange for medium, green for low
- **Responsive Layout**: Multi-column cards with metrics

## 🔧 Technologies Used
- **Python 3.11**
- **Streamlit**: Web framework
- **Scikit-learn**: Isolation Forest anomaly detection
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization

## 📈 Detection Algorithm
**Isolation Forest** (IsolationForest from scikit-learn):
- Non-parametric outlier detection
- Detects anomalies by isolation rather than profiling
- Parameters:
  - n_estimators: Number of trees (default: 100)
  - contamination: Expected anomaly rate (default: 5%)
  - random_state: 42 (reproducible results)

## 📊 Dataset Overview
**217,441 transactions** with features:
- Timestamp (Jan 1 - May 31, 2023)
- TransactionID (TXN1-TXN1999)
- AccountID (ACC1-ACC15)
- Amount ($0-$99,999)
- Merchant (MerchantA-MerchantJ)
- TransactionType (Transfer, Purchase, Withdrawal)
- Location (London, Tokyo, New York, Los Angeles, San Francisco)

## 🎯 Key Results
- **Total Transactions Analyzed**: 217,441
- **Anomalies Detected**: 10,872 (5.00%)
- **Top Anomalous Locations**: Los Angeles, Tokyo
- **Suspicious Amount**: $543M+

## 🔍 Using the Web Application

### Step 1: Configure Analysis
- Adjust estimators and contamination rate in sidebar
- Select date range for analysis
- Default settings analyze 2023 Q1-Q2 data

### Step 2: View Metrics
- See dashboard metrics automatically update
- Risk rate color changes based on anomaly percentage
- Risk ratings: High (>10%), Medium (5-10%), Low (<5%)

### Step 3: Explore Anomalies
- Click "Anomalies" tab to view suspicious transactions
- Sort by AnomalyScore to see most suspicious first
- View anomaly distribution charts

### Step 4: Analyze Data
- "Analytics" tab shows overall patterns
- "Detailed View" allows filtering by type, merchant, anomaly status
- Download filtered results as CSV

## 💾 Environment Conda
```yaml
name: financial-anomaly
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - streamlit
  - pip
```

## ✅ Verification
Run this to verify all is working:
```powershell
conda activate financial-anomaly
python -c "import streamlit, pandas, sklearn, matplotlib, seaborn; print('✅ All packages installed')"
```

## 📝 Notes
- First run may take 10-15 seconds to load all data and train model
- App caches the model after first run for faster interaction
- All charts are generated with purple theme for consistency
- Anomaly detection runs on-demand when parameters change

## 🚀 Performance
- **Data Loading**: < 1 second
- **Model Training**: 3-5 seconds (Isolation Forest with 100 estimators)
- **Analysis**: < 2 seconds
- **Total Runtime**: ~5-8 seconds on first load

## 📧 Troubleshooting

### Port 8501 Already in Use
```powershell
streamlit run app.py --server.port 8502
```

### Slow Performance
- Reduce number of estimators (50-100)
- Use smaller date range
- Close other applications

### Module Not Found
```powershell
conda activate financial-anomaly
pip install -r requirements.txt
```

---

**Created**: June 1, 2026
**Status**: ✅ Complete and Running
**Access**: http://localhost:8501
