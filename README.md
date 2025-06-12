# 🧠 Sentiment-Driven Dynamic Pricing Optimization

> 📈 Leverage customer reviews and ratings to **dynamically recommend the optimal product price** using machine learning, sentiment analysis, and elasticity modeling.

---

## 🚀 Overview

In today's competitive e-commerce and retail landscape, pricing decisions significantly influence customer perception, satisfaction, and revenue. Traditionally, prices are static or manually tweaked, missing out on real-time customer feedback. This project introduces a smarter way to determine product pricing using **sentiment-aware intelligence**.

The application integrates **Natural Language Processing (NLP)**, **machine learning models**, and **customer segmentation** techniques to analyze a product’s reviews and ratings. It then recommends an optimal price that balances **customer sentiment** with **profit potential**. It even predicts how changes in pricing might affect customer perception — a powerful insight for decision-makers.

✅ Built for **retailers, e-commerce analysts, and marketers**, this tool transforms unstructured customer feedback into actionable pricing strategies — all within a clean, interactive Streamlit interface.

---

## 📌 Table of Contents

| Section | Description |
|--------|-------------|
| [Features](#-key-features) | Core functionality of the app |
| [Unique Highlights](#-what-makes-it-stand-out) | Unique selling points |
| [App Structure](#-app-structure) | Code and logic breakdown |

---

## ✅ Key Features

| Feature | Description |
|--------|-------------|
| 💬 **Sentiment Analysis** | Uses TextBlob to evaluate polarity from user review text |
| ⭐ **Rating Input** | Accepts user-supplied star ratings (1–5) |
| 📈 **Price Prediction** | Uses Random Forest to predict an optimal price |
| 👥 **Customer Segmentation** | Clusters customers based on sentiment, length, and rating using KMeans |
| 🔁 **Elasticity Modeling** | Predicts how a price change may shift customer sentiment |
| 📥 **Downloadable Report** | Provides a full summary of inputs and predictions in `.txt` format |
| 📊 **Interactive Elasticity Graph** | Plots price vs. predicted sentiment using Seaborn |

---

## 🌟 What Makes It Stand Out

This isn't just another price prediction tool. Here's what sets it apart:

- ✅ **Multi-Model Integration**: Combines NLP + Clustering + Regression + Tree-based Models.
- 🧠 **Smart Price Ranges**: Displays safe price range using confidence intervals across multiple trees.
- 📉 **Sentiment Elasticity Insight**: Predicts how customer sentiment shifts with price — a rare feature.
- 🧾 **Auto-Generated Reports**: One-click download of a pricing summary for business analysis.
- ⚡ **Streamlit UI**: Interactive, responsive, and highly visual.

These features make it ideal for businesses looking to **maximize ROI** and **enhance customer satisfaction** using intelligent pricing techniques.

---

## 🧱 App Structure

| Component | Role |
|----------|------|
| `streamlit_app.py` | Main UI logic for the app |
| `price_predictor.pkl` | Trained Random Forest model for price prediction |
| `kmeans_model.pkl` | KMeans model for customer segmentation |
| `elasticity.pkl` | Linear model to predict sentiment change due to price |
| `sample_reviews.csv` | Dataset used for testing |
| `pricing_sample_report.txt` | Sample generated report |
