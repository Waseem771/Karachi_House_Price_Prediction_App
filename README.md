# 🏠 Karachi House Price Predictor

An interactive Streamlit web app that predicts real estate prices in Karachi, Pakistan, using a Random Forest Regressor trained on 60,000+ property listings.

🔗 **Live App:** https://karachi-house-price-predictor-app.streamlit.app/
💻 **Source Code:** https://github.com/Waseem771/Karachi_House_Price_Prediction_App

---

## 📋 Overview

This app takes property details — location, size, bedrooms, bathrooms, property type, and purpose (sale/rent) — and returns an instant estimated price, along with market context to help evaluate whether the estimate is reasonable.

**Model performance:** R² ≈ 0.91 on held-out test data.

---

## ✨ Features

- **🎯 Price Prediction** — Enter property details and get an instant estimated price with price-per-sqft breakdown
- **📊 Market Analysis** — Visualize average prices by area, property type distribution, price distribution, and bedroom/size correlations
- **📈 Dataset Insights** — Summary statistics, top listing areas, and distribution breakdowns for bedrooms, bathrooms, and area
- **❓ Help & Guide** — Built-in usage instructions and FAQs
- **🏘️ Market Comparison** — Compares your prediction against similar properties actually in the dataset

---

## 🧠 How It Works

1. **Data source:** Property listings scraped from Zameen.com, filtered to Karachi only
2. **Cleaning:** Duplicates removed, missing prices dropped, outliers trimmed (1st–99th percentile), missing numeric fields filled with medians
3. **Feature engineering:** Top 15 most common locations are kept individually; all others are grouped as `Other`. Categorical features (`location`, `property_type`, `purpose`) are one-hot encoded
4. **Model:** `RandomForestRegressor` (100 trees, max depth 15) trained via an 80/20 train-test split
5. **Caching:** Data loading is cached with `st.cache_data`, and the trained model is cached with `st.cache_resource`, so the app doesn't reload/retrain on every interaction

---

## 🛠️ Tech Stack

| Component         | Library                              |
| ----------------- | ------------------------------------ |
| Web app framework | Streamlit                            |
| Data handling     | pandas, numpy                        |
| ML model          | scikit-learn (RandomForestRegressor) |
| Visualizations    | Plotly, matplotlib, seaborn          |
| Model persistence | joblib                               |

See [`requirements.txt`](./requirements.txt) for exact package versions.

---

## 📁 Project Structure

```
Karachi_House_Price_Prediction_App/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── data/
    └── Property_with_Feature_Engineering.csv   # Dataset (60,000+ Karachi properties)
```

---

## 🚀 Running Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/Waseem771/Karachi_House_Price_Prediction_App.git
   cd Karachi_House_Price_Prediction_App
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints in your browser (usually `http://localhost:8501`)

> **Note:** The app loads its dataset directly from this GitHub repo's `data/` folder at runtime, so an internet connection is required. If you're working offline, place a local copy of `Property_with_Feature_Engineering.csv` in a `data/` folder next to `app.py` — the app will fall back to that automatically.

---

## 📊 Dataset

- **Source:** [Zameen.com](https://www.zameen.com) property listings (via Kaggle)
- **Size:** ~191,000 raw listings, ~59,000 after cleaning and filtering to Karachi
- **Key columns used:** `price`, `location`, `city`, `property_type`, `purpose`, `bedrooms`, `baths`, `area_sqft`, `year`

---

## ⚠️ Disclaimer

Predictions are estimates based on historical listing data and are intended for informational purposes only. Actual market prices can vary significantly due to factors not captured in this dataset (condition, amenities, exact street, current market sentiment, etc.). Always cross-check with current listings before making real financial decisions.

---

---

## 👤 Author

**Waseem Hassan**

- GitHub: [@Waseem771](https://github.com/Waseem771)
- LinkedIn: [waseemhassanshk](https://linkedin.com/in/waseemhassanshk)
- Kaggle: [waseem7711](https://kaggle.com/waseem7711)
