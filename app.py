import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="💰 Hệ thống phát hiện bất thường tài chính💰",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
#thiết lập màu trang web
st.markdown("""
<style>

/* Nền toàn bộ trang */
.stApp {
    background-color: #0F172A !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
}

/* Card thống kê */
.metric-card {
    background-color: #1E293B;
    padding: 22px;
    border-radius: 14px;
    border-left: 6px solid #38BDF8;
    margin: 10px 0;
    box-shadow: 0px 4px 15px rgba(56, 189, 248, 0.2);
}

/* Tiêu đề chính */
.header {
    color: #FFFFFF;
    text-align: center;
    font-weight: bold;
    font-size: 34px;
    margin-bottom: 25px;
}

/* Tiêu đề phụ */
.subheader {
    color: #38BDF8;
    font-weight: bold;
    font-size: 22px;
}

/* Chữ trong sidebar */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Chữ thông thường */
p, h1, h2, h3, h4, h5, h6, label, span, div {
    color: white;
}

</style>
""", unsafe_allow_html=True)
# Title
st.markdown('<div class="header">💰 Hệ thống phát hiện bất thường tài chính</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="subheader">⚙️ Cấu hình</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Load data
REQUIRED_COLUMNS = ["Timestamp", "TransactionID", "AccountID", "Amount", "Merchant", "TransactionType", "Location"]

@st.cache_data
def load_default_data():
    df = pd.read_csv("financial_anomaly_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.dropna(subset=["Timestamp", "Amount"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df

def load_uploaded_data(uploaded_file):
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        
        # Verify columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Tệp thiếu các cột bắt buộc: {', '.join(missing_cols)}")
            
        # Parse Timestamp safely
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        if df["Timestamp"].isnull().all():
            raise ValueError("Không thể phân tích cột 'Timestamp' thành kiểu ngày tháng. Vui lòng kiểm tra định dạng dữ liệu ngày tháng.")
        
        # Drop rows with invalid Timestamp or Amount
        df = df.dropna(subset=["Timestamp", "Amount"])
        if len(df) == 0:
            raise ValueError("Không có dữ liệu hợp lệ (ngày tháng hoặc số tiền không bị trống) sau khi lọc.")
            
        # Sort by Timestamp
        df = df.sort_values("Timestamp").reset_index(drop=True)
        return df
    except Exception as exc:
        raise exc

# Download sample template
@st.cache_data
def get_sample_template():
    try:
        sample_df = pd.read_csv("financial_anomaly_data.csv", nrows=5)
        return sample_df.to_csv(index=False).encode('utf-8')
    except Exception:
        # Fallback empty structure
        fallback_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        return fallback_df.to_csv(index=False).encode('utf-8')

st.sidebar.markdown('<div class="subheader">📥 Tải mẫu dữ liệu</div>', unsafe_allow_html=True)
sample_csv = get_sample_template()
st.sidebar.download_button(
    label="📥 Tải tệp CSV mẫu",
    data=sample_csv,
    file_name="mau_giao_dich_tai_chinh.csv",
    mime="text/csv",
    help="Tải xuống tệp mẫu cấu trúc cột hợp lệ"
)
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Tải lên tệp dữ liệu giao dịch (.csv)",
    type=["csv"],
    help="Tải lên một tệp CSV chứa cột Timestamp, Amount, TransactionType, Location, Merchant"
)

if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        st.sidebar.success("Đã tải lên dữ liệu thành công.")
    except Exception as exc:
        st.sidebar.error(f"Lỗi đọc dữ liệu: {exc}")
        st.stop()
else:
    try:
        df = load_default_data()
    except FileNotFoundError:
        st.sidebar.error("Không tìm thấy tệp dữ liệu mặc định financial_anomaly_data.csv.")
        st.stop()

# Display active data file info
if uploaded_file is not None:
    st.info(f"📂 **Đang phân tích tệp dữ liệu đã tải lên:** `{uploaded_file.name}` (Có {len(df):,} giao dịch hợp lệ)")
else:
    st.info(f"ℹ️ **Đang sử dụng dữ liệu mẫu mặc định:** `financial_anomaly_data.csv` (Có {len(df):,} giao dịch)")

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.day_name()
    df["IsHighAmount"] = df["Amount"] > df["Amount"].quantile(0.95)
    return df

# Anomaly detection
@st.cache_data
def detect_anomalies(df, n_estimators=100, contamination=0.05):
    df = preprocess_data(df)
    features = pd.get_dummies(
        df[["Amount", "Hour", "TransactionType", "Location", "Merchant"]], 
        drop_first=True
    )
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(features)
    df["AnomalyScore"] = model.decision_function(features)
    predictions = model.predict(features)
    df["Anomaly"] = (predictions == -1).astype(int)
    return df

# Sidebar inputs
st.sidebar.subheader("🎯 Tham số phân tích")
n_estimators = st.sidebar.slider(
    "Số lượng cây (estimators):",
    min_value=50,
    max_value=300,
    value=100,
    step=50,
    help="Số cây lớn hơn sẽ tăng độ chính xác nhưng chậm hơn"
)

contamination = st.sidebar.slider(
    "Tỉ lệ dị thường (%):",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Tỉ lệ dự kiến của các giao dịch bất thường"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Lọc theo khoảng thời gian")

# Generate dynamic key based on file to prevent out of range validation errors
file_suffix = "default" if uploaded_file is None else f"uploaded_{uploaded_file.name}"

min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()

date_from = st.sidebar.date_input(
    "Từ ngày:",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    key=f"date_from_{file_suffix}"
)

date_to = st.sidebar.date_input(
    "Đến ngày:",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key=f"date_to_{file_suffix}"
)

# Filter data by date
df_filtered = df[(df["Timestamp"].dt.date >= date_from) & (df["Timestamp"].dt.date <= date_to)].copy()

# Run detection
with st.spinner("🔍 Đang phân tích giao dịch..."):
    df_analyzed = detect_anomalies(df_filtered, n_estimators=n_estimators, contamination=contamination/100)

# Calculate metrics
total_transactions = len(df_analyzed)
anomaly_transactions = len(df_analyzed[df_analyzed["Anomaly"] == 1])
anomaly_rate = (anomaly_transactions / total_transactions * 100) if total_transactions > 0 else 0
suspicious_amount = df_analyzed[df_analyzed["Anomaly"] == 1]["Amount"].sum()

# Create metrics columns
st.markdown('<div class="subheader">📈 THỐNG KÊ CHÍNH</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #FFFFFF;">📊 Tổng số giao dịch</h3>
        <h2 style="margin: 5px 0; color: #FFFFFF;">{total_transactions:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #FFFFFF;">⚠️ Giao dịch bất thường</h3>
        <h2 style="margin: 5px 0; color: #d32f2f;">{anomaly_transactions:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    risk_color = "#d32f2f" if anomaly_rate > 10 else "#f57c00" if anomaly_rate > 5 else "#388e3c"
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #FFFFFF;">📊 Tỷ lệ rủi ro</h3>
        <h2 style="margin: 5px 0; color: {risk_color};">{anomaly_rate:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #FFFFFF;">💸 Số tiền nghi ngờ</h3>
        <h2 style="margin: 5px 0; color: #d32f2f;">${suspicious_amount:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan", "🔴 Bất thường", "📈 Phân tích", "🔍 Chi tiết"])

# Tab 1: Overview
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subheader">Phân bố loại giao dịch</div>', unsafe_allow_html=True)
        type_counts = df_analyzed["TransactionType"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.Purples(np.linspace(0.4, 0.8, len(type_counts)))
        type_counts.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Loại giao dịch", fontsize=14, fontweight="bold", color="#FFFFFF")
        ax.set_ylabel("Số lượng", fontsize=11, color="#FFFFFF")
        ax.set_xlabel("Loại", fontsize=11, color="#FFFFFF")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="subheader">Top 5 địa điểm</div>', unsafe_allow_html=True)
        location_counts = df_analyzed["Location"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.Purples(np.linspace(0.4, 0.8, len(location_counts)))
        location_counts.plot(kind="barh", ax=ax, color=colors)
        ax.set_title("Top 5 địa điểm", fontsize=14, fontweight="bold", color="#FFFFFF")
        ax.set_xlabel("Số lượng", fontsize=11, color="#FFFFFF")
        plt.tight_layout()
        st.pyplot(fig)

# Tab 2: Anomalies Details
with tab2:
    st.markdown('<div class="subheader">🔴 Giao dịch bất thường đã phát hiện</div>', unsafe_allow_html=True)
    
    anomalies_df = df_analyzed[df_analyzed["Anomaly"] == 1].sort_values("AnomalyScore").copy()
    
    if len(anomalies_df) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Top {min(10, len(anomalies_df))} giao dịch bất thường nhất")
        
        with col2:
            rows_to_show = st.selectbox("Hiển thị số dòng:", [5, 10, 20, 50], index=1)
        
        display_cols = ["Timestamp", "TransactionID", "Amount", "Merchant", "TransactionType", "Location", "AnomalyScore"]
        st.dataframe(
            anomalies_df[display_cols].head(rows_to_show),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subheader">Phân bố giá trị giao dịch bất thường</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(anomalies_df["Amount"], bins=30, color="#ce93d8", edgecolor="#8e24aa")
            ax.set_title("Giá trị giao dịch bất thường", fontsize=14, fontweight="bold", color="#FFFFFF")
            ax.set_xlabel("Giá trị ($)", fontsize=11, color="#FFFFFF")
            ax.set_ylabel("Tần suất", fontsize=11, color="#FFFFFF")
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown('<div class="subheader">Loại giao dịch bất thường</div>', unsafe_allow_html=True)
            anomaly_types = anomalies_df["TransactionType"].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = plt.cm.Purples(np.linspace(0.4, 0.8, len(anomaly_types)))
            anomaly_types.plot(kind="pie", ax=ax, autopct="%1.1f%%", colors=colors)
            ax.set_title("Các loại bất thường", fontsize=14, fontweight="bold", color="#FFFFFF")
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("✅ Không có giao dịch bất thường trong khoảng thời gian đã chọn!")

# Tab 3: Analytics
with tab3:
    st.markdown('<div class="subheader">📊 Phân tích chi tiết</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subheader">Phân bố giá trị giao dịch (tất cả)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_analyzed["Amount"], bins=50, color="#b39ddb", edgecolor="#FFFFFF", alpha=0.7)
        ax.set_title("Phân bố giá trị giao dịch", fontsize=14, fontweight="bold", color="#FFFFFF")
        ax.set_xlabel("Giá trị ($)", fontsize=11, color="#FFFFFF")
        ax.set_ylabel("Tần suất", fontsize=11, color="#FFFFFF")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="subheader">Giá trị theo loại giao dịch</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        df_analyzed.boxplot(column="Amount", by="TransactionType", ax=ax)
        ax.set_title("Giá trị theo loại giao dịch", fontsize=14, fontweight="bold", color="#FFFFFF")
        ax.set_ylabel("Giá trị ($)", fontsize=11, color="#FFFFFF")
        ax.set_xlabel("Loại giao dịch", fontsize=11, color="#FFFFFF")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)

# Tab 4: Detailed Data View
with tab4:
    st.markdown('<div class="subheader">📋 Xem chi tiết giao dịch</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox("Lọc theo loại:", ["Tất cả"] + df_analyzed["TransactionType"].unique().tolist())
    
    with col2:
        filter_merchant = st.selectbox("Lọc theo nhà cung cấp:", ["Tất cả"] + df_analyzed["Merchant"].unique().tolist())
    
    with col3:
        filter_anomaly = st.selectbox("Hiển thị:", ["Tất cả", "Chỉ bất thường", "Chỉ bình thường"])
    
    # Apply filters
    filtered_data = df_analyzed.copy()
    
    if filter_type != "Tất cả":
        filtered_data = filtered_data[filtered_data["TransactionType"] == filter_type]
    
    if filter_merchant != "Tất cả":
        filtered_data = filtered_data[filtered_data["Merchant"] == filter_merchant]
    
    if filter_anomaly == "Chỉ bất thường":
        filtered_data = filtered_data[filtered_data["Anomaly"] == 1]
    elif filter_anomaly == "Chỉ bình thường":
        filtered_data = filtered_data[filtered_data["Anomaly"] == 0]
    
    # Display table
    display_cols = ["Timestamp", "TransactionID", "AccountID", "Amount", "Merchant", "TransactionType", "Location", "Anomaly", "AnomalyScore"]
    st.dataframe(
        filtered_data[display_cols],
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Tải xuống dữ liệu đã lọc (CSV)",
        data=csv,
        file_name=f"filtered_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8e24aa; font-size: 12px;">
    💰 Hệ thống phát hiện bất thường tài chính | Powered by Streamlit & Scikit-learn
    <br/>
    Cập nhật lần cuối: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
</div>
""", unsafe_allow_html=True)
