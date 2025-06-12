import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Set Streamlit page configuration
st.set_page_config(page_title="🧠 Price Optimizer", layout="centered")

# Load models
rf = joblib.load("models/price_predictor.pkl")
kmeans = joblib.load("models/kmeans_model.pkl")
lr = joblib.load("models/elasticity.pkl")

# Sidebar
st.sidebar.title("📂 About this app")
st.sidebar.markdown(
    '''
    This app analyzes customer **reviews** and **ratings** to recommend an optimal product **price** using:
- 💬 Sentiment Analysis  
- 🧠 Clustering (KMeans)  
- 📈 Elasticity Modeling  
- 📊 Random Forest for Price Prediction 
    '''
)
st.sidebar.markdown("Built with Python, Streamlit, and ML")

# User inputs
review = st.text_area("✍️ Enter Customer Review: ", height=150)
rating = st.slider("⭐ Rating (1 to 5)", 1, 5)
current_price = st.number_input("💰 Current Product Price", value=100.0)

# Tabs for outputs
tab1, tab2 = st.tabs(["🔍 Analysis", "📉 Elasticity Chart"])

# On button click
if st.button("🚀 Analyze & Recommend"):
    # Sentiment Analysis
    polarity = TextBlob(review).sentiment.polarity
    review_len = len(review)
    sentiment_code = 0 if polarity < -0.1 else 1 if polarity < 0.1 else 2

    # Clustering
    cluster_id = kmeans.predict([[rating, polarity, review_len, sentiment_code]])[0]

    # Price Prediction
    input_vec = np.array([[rating, polarity, review_len, cluster_id]])
    predicted_price = rf.predict(input_vec)[0]
    all_preds = [tree.predict(input_vec)[0] for tree in rf.estimators_]
    low_price = np.percentile(all_preds, 10)
    high_price = np.percentile(all_preds, 90)

    # Elasticity (Sentiment Shift Estimate)
    sentiment_change = lr.coef_[0] * (predicted_price - current_price)
    sentiment_change_value = sentiment_change.item()

    # Output in Tab 1
    with tab1:
        st.markdown(f"### 💡 Recommended Price: ₹ **{predicted_price:.2f}**")
        st.markdown(f"### 📉 Safe Price Range: ₹ {low_price:.2f} - ₹ {high_price:.2f}")
        st.markdown(f"### 👥 Customer Cluster: **{cluster_id}**")
        st.markdown("### 💬 Review Emotion:")
        if polarity < -0.2:
            st.error("Negative 😔")
        elif polarity < 0.2:
            st.warning("Neutral 😐")
        else:
            st.success("Positive 😊")

        st.markdown(f"### 🔁 Expected Sentiment Shift: **{sentiment_change_value:.3f}**")

    # Generate downloadable report
    report_text = f'''
🔍 Review Analysis Report

💬 Review: {review}
⭐ Rating: {rating}
💰 Current Price: ₹{current_price:.2f}

📊 Predicted Optimal Price: ₹{predicted_price:.2f}
📉 Safe Price Range: ₹{low_price:.2f} - ₹{high_price:.2f}
👥 Customer Cluster: {cluster_id}
💬 Sentiment: {'Negative 😔' if polarity < -0.1 else 'Neutral 😐' if polarity < 0.1 else 'Positive 😊'}
🔁 Expected Sentiment Shift: {sentiment_change_value:.3f}
    '''

    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name="pricing_report.txt",
        mime="text/plain"
    )

    # Show Elasticity Chart in Tab 2
    with tab2:
        st.markdown("### 📊 Sentiment vs Price (Elasticity Curve)")
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)
        predicted_sentiments = lr.predict(price_range.reshape(-1, 1)).flatten()

        df_elastic = pd.DataFrame({
            "Price (₹)": price_range,
            "Predicted Sentiment Polarity": predicted_sentiments
        })

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_elastic, x="Price (₹)", y="Predicted Sentiment Polarity", marker='o')
        plt.title("📉 Sentiment Change with Price")
        plt.xlabel("Price (₹)")
        plt.ylabel("Predicted Sentiment")
        st.pyplot(plt)
