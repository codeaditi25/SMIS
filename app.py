import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="VIT Smart Mess Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* main background */
    .stApp { background-color: #0f0f0f; color: white; }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 2px solid #e94560;
    }

    /* metric cards */
    div[data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #533483;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(233,69,96,0.2);
    }

    /* buttons */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #533483);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 15px;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #533483, #e94560);
        transform: scale(1.02);
    }

    /* title */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e94560, #ffd166, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 30px;
    }

    /* section headers */
    .section-header {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-left: 4px solid #e94560;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        font-size: 1.2rem;
        font-weight: bold;
    }

    /* prediction result box */
    .pred-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 2px solid #06d6a0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .pred-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #06d6a0;
    }

    /* advisory boxes */
    .advisory-safe {
        background: linear-gradient(135deg, #0d3d2e, #1a6b4a);
        border: 2px solid #06d6a0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .advisory-warning {
        background: linear-gradient(135deg, #3d3000, #6b5500);
        border: 2px solid #ffd166;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .advisory-danger {
        background: linear-gradient(135deg, #3d0d1a, #6b1530);
        border: 2px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── CHART STYLE ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a2e",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.5,
    "grid.linestyle":   "--",
})

# ── LOAD DATA & MODELS ─────────────────────────────────
@st.cache_data
def load_data():
    footfall = pd.read_csv("dataset/footfall_data.csv")
    feedback = pd.read_csv("dataset/dish_feedback.csv")
    waste    = pd.read_csv("dataset/waste_data.csv")
    footfall["date"] = pd.to_datetime(footfall["date"])
    waste["date"]    = pd.to_datetime(waste["date"])
    feedback["date"] = pd.to_datetime(feedback["date"])
    return footfall, feedback, waste

@st.cache_resource
def load_models():
    rf1        = joblib.load("models/footfall_model.pkl")
    rf2        = joblib.load("models/waste_model.pkl")
    rf3        = joblib.load("models/sentiment_model.pkl")
    le_day     = joblib.load("models/le_day.pkl")
    le_weather = joblib.load("models/le_weather.pkl")
    le_dish    = joblib.load("models/le_dish.pkl")
    le_sent    = joblib.load("models/le_sent.pkl")
    return rf1, rf2, rf3, le_day, le_weather, le_dish, le_sent

footfall, feedback, waste = load_data()
rf1, rf2, rf3, le_day, le_weather, le_dish, le_sent = load_models()

# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0'>
        <div style='font-size:3rem'>🍽️</div>
        <div style='font-size:1.2rem; font-weight:900;
                    color:#e94560'>VIT Smart Mess</div>
        <div style='font-size:0.8rem; color:#888'>
            Intelligence System</div>
    </div>
    <hr style='border-color:#e94560'>
    """, unsafe_allow_html=True)

    page = st.radio("📌 Navigate", [
        "🏠 Dashboard",
        "🤖 Live Predictions",
        "📊 Analytics",
        "😊 Sentiment Analysis",
        "💡 Insights & Savings"
    ])

    st.markdown("<hr style='border-color:#533483'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#666;
                text-align:center; padding:10px'>
        Built with ❤️ for VIT Bhopal<br>
        CSA2001 — AI & ML Project<br>
        <span style='color:#e94560'>Random Forest + NLP</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="main-title">🍽️ VIT Bhopal Smart Mess Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Mess Management System | CSA2001 Project</div>',
                unsafe_allow_html=True)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Days Tracked",
                  f"{len(footfall)}", "+365 days")
    with col2:
        st.metric("🧑 Avg Daily Students",
                  f"{int(footfall['total'].mean())}",
                  f"Max: {int(footfall['total'].max())}")
    with col3:
        st.metric("🗑️ Total Food Wasted",
                  f"{waste['waste_kg'].sum():.0f} kg",
                  f"Avg: {waste['waste_kg'].mean():.1f} kg/day")
    with col4:
        st.metric("💰 Money Wasted",
                  f"₹{waste['money_wasted_rs'].sum():,.0f}",
                  "Can be saved with AI!")

    st.markdown("---")

    # footfall trend
    st.markdown('<div class="section-header">📈 Daily Footfall Trend</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(footfall["date"], footfall["breakfast"],
            color="#ffd166", linewidth=1.5, label="Breakfast")
    ax.plot(footfall["date"], footfall["lunch"],
            color="#06d6a0", linewidth=1.5, label="Lunch")
    ax.plot(footfall["date"], footfall["dinner"],
            color="#e94560", linewidth=1.5, label="Dinner")
    ax.fill_between(footfall["date"],
                    footfall["lunch"], alpha=0.1, color="#06d6a0")
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Students")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # day avg + sentiment side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📅 Footfall by Day</div>',
                    unsafe_allow_html=True)
        day_order = ["Monday","Tuesday","Wednesday",
                     "Thursday","Friday","Saturday","Sunday"]
        day_avg   = footfall.groupby("day")["total"].mean().reindex(day_order)
        fig, ax   = plt.subplots(figsize=(7, 4))
        colors    = ["#e94560","#06d6a0","#ffd166",
                     "#533483","#118ab2","#ef476f","#073b4c"]
        ax.bar(day_avg.index, day_avg.values,
               color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Avg Students")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">😊 Sentiment Overview</div>',
                    unsafe_allow_html=True)
        sc  = feedback["sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.pie(sc.values,
               labels=sc.index,
               autopct="%1.1f%%",
               colors=["#06d6a0","#ffd166","#e94560"],
               startangle=140,
               textprops={"color":"white","fontsize":12},
               wedgeprops={"edgecolor":"#0f0f0f","linewidth":2})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════
# PAGE 2 — LIVE PREDICTIONS
# ══════════════════════════════════════════════════════
elif page == "🤖 Live Predictions":
    st.markdown('<div class="main-title">🤖 Live AI Predictions</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter today\'s parameters and get instant predictions</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">⚙️ Input Parameters</div>',
                    unsafe_allow_html=True)
        day     = st.selectbox("📅 Day of Week", [
            "Monday","Tuesday","Wednesday",
            "Thursday","Friday","Saturday","Sunday"])
        weather = st.selectbox("🌤️ Weather", [
            "Pleasant","Hot","Rainy","Cold"])
        exam    = st.checkbox("📝 Exam Week?")
        holiday = st.checkbox("🎉 Holiday?")

        predict_btn = st.button("🚀 Predict Now!")

    with col2:
        st.markdown('<div class="section-header">📊 Prediction Results</div>',
                    unsafe_allow_html=True)

        if predict_btn:
            # encode inputs
            try:
                day_enc     = le_day.transform([day])[0]
                weather_enc = le_weather.transform([weather])[0]
            except:
                day_enc     = 0
                weather_enc = 0

            exam_enc    = int(exam)
            holiday_enc = int(holiday)

            # footfall prediction
            X_foot = np.array([[day_enc, weather_enc,
                                 exam_enc, holiday_enc]])
            pred_foot = int(rf1.predict(X_foot)[0])

            # waste prediction
            food_kg  = round(pred_foot * 0.35, 2)
            X_waste  = np.array([[day_enc, weather_enc,
                                   pred_foot, food_kg]])
            pred_waste = round(rf2.predict(X_waste)[0], 2)
            pred_money = round(pred_waste * 80, 2)

            # advisory level
            waste_pct = (pred_waste / food_kg) * 100
            if waste_pct < 10:
                advisory = "🟢 LOW WASTE"
                adv_class = "advisory-safe"
                adv_msg   = "Excellent! Mess is operating efficiently."
            elif waste_pct < 16:
                advisory = "🟡 MODERATE WASTE"
                adv_class = "advisory-warning"
                adv_msg   = "Acceptable. Consider reducing portions slightly."
            else:
                advisory = "🔴 HIGH WASTE ALERT"
                adv_class = "advisory-danger"
                adv_msg   = "Reduce food preparation by 15-20% today!"

            # display results
            st.markdown(f"""
            <div class="pred-box">
                <div style='color:#888; font-size:0.9rem'>
                    Predicted Students Today</div>
                <div class="pred-value">{pred_foot:,}</div>
                <div style='color:#888'>students expected</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="pred-box" style='border-color:#e94560; margin-top:10px'>
                <div style='color:#888; font-size:0.9rem'>
                    Predicted Food Waste</div>
                <div class="pred-value" style='color:#e94560'>
                    {pred_waste} kg</div>
                <div style='color:#888'>₹{pred_money} money at risk</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="{adv_class}" style='margin-top:10px'>
                <div style='font-size:1.3rem; font-weight:900'>
                    {advisory}</div>
                <div style='margin-top:8px; font-size:0.95rem'>
                    {adv_msg}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("👈 Set parameters and click **Predict Now!**")

# ══════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown('<div class="main-title">📊 Deep Analytics</div>',
                unsafe_allow_html=True)

    # heatmap
    st.markdown('<div class="section-header">🌡️ Footfall Heatmap: Day vs Weather</div>',
                unsafe_allow_html=True)
    day_order  = ["Monday","Tuesday","Wednesday",
                  "Thursday","Friday","Saturday","Sunday"]
    heat_data  = footfall.groupby(
        ["day","weather"])["total"].mean().unstack()
    heat_data  = heat_data.reindex(day_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat_data, annot=True, fmt=".0f",
                cmap="YlOrRd",
                linewidths=0.5,
                linecolor="#0f0f0f",
                ax=ax,
                annot_kws={"size":12,"weight":"bold"})
    ax.set_xlabel("Weather")
    ax.set_ylabel("Day")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # monthly waste
    st.markdown('<div class="section-header">📉 Monthly Waste Trend</div>',
                unsafe_allow_html=True)
    waste["month"]    = waste["date"].dt.to_period("M")
    monthly_waste     = waste.groupby("month")["waste_kg"].sum()
    monthly_money     = waste.groupby("month")["money_wasted_rs"].sum()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2      = ax1.twinx()
    ax1.bar([str(m) for m in monthly_waste.index],
            monthly_waste.values,
            color="#533483", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax2.plot([str(m) for m in monthly_money.index],
             monthly_money.values,
             color="#ffd166", linewidth=2.5,
             marker="o", markersize=6)
    ax1.set_ylabel("Waste (kg)", color="#533483")
    ax2.set_ylabel("Money (₹)", color="#ffd166")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════
# PAGE 4 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "😊 Sentiment Analysis":
    st.markdown('<div class="main-title">😊 Meal Sentiment Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # dish popularity
        st.markdown('<div class="section-header">🏆 Most Loved Dishes</div>',
                    unsafe_allow_html=True)
        dish_sent = feedback.groupby(
            ["dish","sentiment"])["votes"].sum().unstack(fill_value=0)
        if "Positive" in dish_sent.columns:
            top_loved = dish_sent["Positive"].nlargest(6)
            fig, ax   = plt.subplots(figsize=(7, 5))
            ax.barh(top_loved.index, top_loved.values,
                    color="#06d6a0", edgecolor="white")
            ax.set_xlabel("Positive Votes")
            ax.grid(True, axis="x")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown('<div class="section-header">👎 Most Disliked Dishes</div>',
                    unsafe_allow_html=True)
        if "Negative" in dish_sent.columns:
            top_hated = dish_sent["Negative"].nlargest(6)
            fig, ax   = plt.subplots(figsize=(7, 5))
            ax.barh(top_hated.index, top_hated.values,
                    color="#e94560", edgecolor="white")
            ax.set_xlabel("Negative Votes")
            ax.grid(True, axis="x")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # avg rating per dish
    st.markdown('<div class="section-header">⭐ Average Rating per Dish</div>',
                unsafe_allow_html=True)
    avg_rating = feedback.groupby("dish")["rating"].mean().sort_values()
    fig, ax    = plt.subplots(figsize=(12, 5))
    bars       = ax.bar(avg_rating.index, avg_rating.values,
                        color=["#e94560" if v < 3 else
                               "#ffd166" if v < 4 else
                               "#06d6a0" for v in avg_rating.values],
                        edgecolor="white", linewidth=0.5)
    ax.axhline(y=3, color="#ffd166",
               linestyle="--", linewidth=1.5,
               label="Acceptable threshold")
    ax.set_ylabel("Average Rating (1-5)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()