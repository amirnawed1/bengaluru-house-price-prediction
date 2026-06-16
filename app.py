import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ========== HEADER SECTION ==========
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://img.icons8.com/fluency/96/home.png", width=100)
with col2:
    st.title("🏠 Bengaluru House Price Predictor")
    st.markdown("### Estimate property prices based on location, size & configuration")
    st.markdown("---")

# Load data and model
@st.cache_resource
def load_model():
    return pickle.load(open("RidgeModel.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

data = load_data()
model = load_model()

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📋 Enter Property Details")
    st.markdown("---")

    location = st.selectbox(
        "📍 Location",
        sorted(data["location"].unique())
    )

    st.markdown("---")

    sqft_option = st.selectbox(
        "📐 Total Square Feet",
        ["500-750", "750-1000", "1000-1250", "1250-1500", "1500-2000", "2000-3000", "3000+"],
        index=2
    )

    sqft_map = {
        "500-750": 625,
        "750-1000": 875,
        "1000-1250": 1125,
        "1250-1500": 1375,
        "1500-2000": 1750,
        "2000-3000": 2500,
        "3000+": 3500
    }
    total_sqft = sqft_map[sqft_option]

    st.markdown("---")

    bhk = st.selectbox(
        "🛏️ BHK",
        [1, 2, 3, 4, 5],
        index=1
    )

    bath = st.selectbox(
        "🛁 Bathrooms",
        [1, 2, 3, 4],
        index=1
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

# ========== MAIN CONTENT AREA ==========
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Price Insights")
    st.info(
        f"""
        **Selected Configuration:**
        - 📍 Location: **{location}**
        - 📐 Area: **{total_sqft:.0f} sq.ft**
        - 🛏️ BHK: **{bhk}**
        - 🛁 Bathrooms: **{bath}**
        """
    )
    
    # Quick stats
    st.markdown("### 📈 Market Overview")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Avg Price (Lakhs)", f"₹{data['price'].mean():.1f}")
    with metric_col2:
        st.metric("Max Price (Lakhs)", f"₹{data['price'].max():.1f}")
    with metric_col3:
        st.metric("Total Listings", f"{len(data):,}")

with col2:
    st.markdown("### 💰 Price Distribution in Bengaluru")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data["price"], bins=50, kde=True, color="#FF6B6B", ax=ax)
    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Number of Properties")
    ax.set_title("Price Distribution Across Bengaluru")
    ax.axvline(data["price"].mean(), color="blue", linestyle="--", label="Average")
    ax.legend()
    st.pyplot(fig)

# ========== PREDICTION RESULT ==========
if predict_btn:
    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")
    
    input_data = pd.DataFrame({
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "bhk": [bhk]
    })
    
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)
    
    # Price per sqft
    price_per_sqft = (prediction * 100000) / total_sqft if total_sqft > 0 else 0
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.success(f"### 🏠 Estimated Price: ₹ {prediction:.2f} Lakhs")
        st.metric("Price in Rupees", f"₹ {prediction * 100000:,.0f}")
        
    with result_col2:
        st.info(f"### 📐 Price per Sq.Ft: ₹ {price_per_sqft:,.0f}")
        
        # Comparison with average
        diff_from_avg = ((prediction - data['price'].mean()) / data['price'].mean()) * 100
        if diff_from_avg > 0:
            st.warning(f"📈 {abs(diff_from_avg):.1f}% above market average")
        else:
            st.success(f"📉 {abs(diff_from_avg):.1f}% below market average")
    
    # Price range indicator
    st.markdown("### 📊 Where Your Price Fits")
    fig, ax = plt.subplots(figsize=(10, 2))
    
    min_price = data["price"].min()
    max_price = data["price"].max()
    
    ax.barh(0, max_price, color="#E0E0E0", height=0.4)
    ax.barh(0, prediction, color="#4ECDC4", height=0.4)
    ax.scatter(prediction, 0, color="#FF6B6B", s=200, zorder=5)
    
    ax.set_xlabel("Price (Lakhs)")
    ax.set_yticks([])
    ax.set_title(f"Your Price vs Market Range (₹{min_price:.0f}L - ₹{max_price:.0f}L)")
    st.pyplot(fig)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("### 👨‍💻 About This Project")
st.markdown("""
- **Model:** Ridge Regression trained on Bengaluru housing data
- **Features:** Location, Square Footage, BHK, Bathrooms
- **Built by:** Amir Nawed | [GitHub](https://github.com/amirnawed1)
- **Deployed on:** Streamlit Cloud
""")
