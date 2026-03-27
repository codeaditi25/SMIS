# 🍽️ VIT Bhopal Smart Mess Intelligence System

> **AI-Powered Mess Management System** — CSA2001: Fundamentals in AI & ML  
> VIT Bhopal University | Academic Year 2024–25

---

## 📌 Problem Statement

College mess facilities across India face three critical inefficiencies daily:
- **Food overproduction** — cooks prepare for maximum capacity regardless of actual footfall
- **Zero feedback loop** — no system to track which dishes students actually like
- **Massive food waste** — thousands of rupees wasted every month with no tracking

This project uses Machine Learning to solve all three problems with real predictive intelligence.

---

## 🎯 What This System Does

| Module | Description |
|--------|-------------|
| 📈 **Footfall Predictor** | Predicts how many students will eat today based on day, weather, exams, and holidays |
| 🗑️ **Food Waste Predictor** | Predicts kilograms of food likely to be wasted and money at risk |
| 😊 **Sentiment Analyzer** | Classifies student feedback as Positive / Neutral / Negative per dish |
| 💡 **Savings Dashboard** | Provides AI-driven actionable recommendations to reduce waste |

---

## 🧠 ML Models Used

| Model | Algorithm | Target | Accuracy |
|-------|-----------|--------|----------|
| Footfall Predictor | Random Forest Regressor | Total daily students | ~94% R² |
| Waste Predictor | Random Forest Regressor | Waste in kg | ~91% R² |
| Sentiment Classifier | Random Forest Classifier | Positive/Neutral/Negative | ~96% Accuracy |

---

## 📊 Course Outcomes Covered

- **CO1** — AI capabilities and intelligent agents solving real problems
- **CO2** — ML algorithms and practical applications
- **CO3** — Statistical analysis, probability, and decision theory
- **CO4** — Classification, Regression, Bias-Variance tradeoff, Overfitting prevention
- **CO5** — NLP sentiment analysis + real-world case study

---

## 🗂️ Project Structure

```
vit-smart-mess/
├── dataset/
│   ├── generate_data.py          ← Synthetic dataset generator
│   ├── footfall_data.csv         ← 365 days of student footfall
│   ├── dish_feedback.csv         ← 1825 dish feedback records
│   └── waste_data.csv            ← 365 days of food waste records
│
├── models/
│   ├── footfall_model.pkl        ← Trained footfall predictor
│   ├── waste_model.pkl           ← Trained waste predictor
│   ├── sentiment_model.pkl       ← Trained sentiment classifier
│   └── le_*.pkl                  ← Label encoders
│
├── notebooks/
│   └── analysis.ipynb            ← EDA and model exploration
│
├── eda.py                        ← Exploratory Data Analysis (6 charts)
├── train_models.py               ← Model training + evaluation
├── app.py                        ← Streamlit web dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/vit-smart-mess-intelligence.git
cd vit-smart-mess-intelligence
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn plotly streamlit joblib
```

### 3. Generate the dataset
```bash
python dataset/generate_data.py
```

### 4. Run EDA and generate charts
```bash
python eda.py
```

### 5. Train all ML models
```bash
python train_models.py
```

### 6. Launch the dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📈 Key Visualizations

- Daily student footfall trend (Breakfast / Lunch / Dinner)
- Average footfall by day of week (bar chart)
- Sentiment distribution pie chart
- Top 5 most loved & most disliked dishes
- Monthly food waste & money lost trend
- Footfall heatmap: Day vs Weather
- Feature importance charts for each ML model
- Actual vs Predicted comparison plots
- Confusion matrix for sentiment classifier

---

## 💰 Impact Analysis (Simulated — VIT Bhopal)

| Metric | Value |
|--------|-------|
| Total food wasted (1 year) | ~18,000 kg |
| Total money wasted | ~₹14,40,000 |
| Potential AI savings (40%) | ~₹5,76,000 |
| Avg waste per day | ~49 kg |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | ML models |
| Matplotlib & Seaborn | Static visualizations |
| Plotly | Interactive charts |
| Streamlit | Web dashboard UI |
| Joblib | Model serialization |

---

## 📁 Dataset Details

All data is **synthetically generated** to simulate realistic VIT Bhopal mess operations. The generator (`generate_data.py`) uses:
- Real academic calendar patterns (exam weeks, holidays)
- Realistic weather cycles for Bhopal (Hot/Rainy/Cold/Pleasant)
- Statistically accurate waste percentages (8–20%)
- Actual mess dishes served at VIT Bhopal

This approach demonstrates **data engineering skills** and makes the project 100% unique.

---

## 🏆 Results Summary

```
Model Performance:
──────────────────────────────────
Footfall Predictor   R²  : ~94%
Waste Predictor      R²  : ~91%
Sentiment Classifier Acc : ~96%
──────────────────────────────────
```

---

## 👨‍💻 Author

**ADITI PANDEY**  
Registration No: 25BCE11306 
B.Tech CSE | Semester 2  
VIT Bhopal University  
Course: CSA2001 — Fundamentals in AI and ML

---

## 📄 License

This project is submitted as an academic assignment for VIT Bhopal University.  
© 2025 — All rights reserved.
