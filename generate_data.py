import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ─── CONFIG ───────────────────────────────────────────
START_DATE = datetime(2023, 6, 1)
END_DATE   = datetime(2024, 5, 31)
TOTAL_STUDENTS = 3000

DISHES = [
    "Idli Sambar", "Poha", "Paratha", "Upma",          # breakfast
    "Dal Rice", "Rajma Rice", "Chole Bhature",           # lunch
    "Roti Sabzi", "Biryani", "Paneer Butter Masala",    # dinner
    "Curd Rice", "Khichdi", "Fried Rice"                # special
]

FEEDBACK_POSITIVE = [
    "food was amazing today!", "loved the biryani!",
    "very tasty and fresh", "excellent quality today",
    "best meal this week", "really enjoyed the food",
    "paneer was delicious", "great taste today"
]

FEEDBACK_NEGATIVE = [
    "food was too salty", "dal was undercooked",
    "very bad quality today", "tasteless food",
    "cold food served again", "not fresh at all",
    "worst meal this month", "rice was overcooked"
]

FEEDBACK_NEUTRAL = [
    "food was okay", "average as usual",
    "nothing special today", "decent meal",
    "could be better", "acceptable quality"
]

# ─── HELPER FUNCTIONS ─────────────────────────────────
def is_exam_week(date):
    # simulate exam weeks in Nov, Dec, Apr, May
    return date.month in [11, 4] and date.day <= 15

def is_holiday(date):
    holidays = [
        datetime(2023, 8, 15), datetime(2023, 10, 2),
        datetime(2023, 11, 14), datetime(2024, 1, 26),
        datetime(2024, 3, 25), datetime(2024, 3, 29),
    ]
    return date in holidays

def get_weather(date):
    month = date.month
    if month in [3, 4, 5]:   return "Hot"
    elif month in [6, 7, 8]: return "Rainy"
    elif month in [11, 12, 1]: return "Cold"
    else:                      return "Pleasant"

# ─── GENERATE FOOTFALL DATA ───────────────────────────
def generate_footfall():
    records = []
    current = START_DATE

    while current <= END_DATE:
        day_name  = current.strftime("%A")
        weather   = get_weather(current)
        exam_week = is_exam_week(current)
        holiday   = is_holiday(current)

        # base footfall logic
        base = TOTAL_STUDENTS * 0.75

        if day_name == "Sunday":    base *= 0.55
        elif day_name == "Saturday": base *= 0.70
        if exam_week:   base *= 0.85
        if holiday:     base *= 0.30
        if weather == "Rainy":    base *= 0.90
        if weather == "Hot":      base *= 0.95

        breakfast = int(base * random.uniform(0.55, 0.70))
        lunch     = int(base * random.uniform(0.75, 0.90))
        dinner    = int(base * random.uniform(0.65, 0.80))

        records.append({
            "date":       current.strftime("%Y-%m-%d"),
            "day":        day_name,
            "weather":    weather,
            "exam_week":  int(exam_week),
            "holiday":    int(holiday),
            "breakfast":  breakfast,
            "lunch":      lunch,
            "dinner":     dinner,
            "total":      breakfast + lunch + dinner
        })

        current += timedelta(days=1)

    df = pd.DataFrame(records)
    df.to_csv("dataset/footfall_data.csv", index=False)
    print(f"✅ footfall_data.csv → {len(df)} rows")

# ─── GENERATE DISH FEEDBACK DATA ──────────────────────
def generate_feedback():
    records = []
    current = START_DATE

    while current <= END_DATE:
        daily_dishes = random.sample(DISHES, k=5)

        for dish in daily_dishes:
            sentiment_roll = random.random()
            if sentiment_roll < 0.55:
                feedback  = random.choice(FEEDBACK_POSITIVE)
                sentiment = "Positive"
                rating    = random.randint(4, 5)
            elif sentiment_roll < 0.80:
                feedback  = random.choice(FEEDBACK_NEUTRAL)
                sentiment = "Neutral"
                rating    = random.randint(3, 3)
            else:
                feedback  = random.choice(FEEDBACK_NEGATIVE)
                sentiment = "Negative"
                rating    = random.randint(1, 2)

            records.append({
                "date":      current.strftime("%Y-%m-%d"),
                "dish":      dish,
                "feedback":  feedback,
                "sentiment": sentiment,
                "rating":    rating,
                "votes":     random.randint(10, 120)
            })

        current += timedelta(days=1)

    df = pd.DataFrame(records)
    df.to_csv("dataset/dish_feedback.csv", index=False)
    print(f"✅ dish_feedback.csv → {len(df)} rows")

# ─── GENERATE FOOD WASTE DATA ─────────────────────────
def generate_waste():
    footfall = pd.read_csv("dataset/footfall_data.csv")
    records  = []

    for _, row in footfall.iterrows():
        # waste logic: more footfall = less waste %
        waste_pct = random.uniform(0.08, 0.20)
        if row["holiday"]:   waste_pct += 0.15
        if row["exam_week"]: waste_pct += 0.05

        total_food_kg   = round(row["total"] * 0.35, 2)  # 350g per student
        waste_kg        = round(total_food_kg * waste_pct, 2)
        cost_per_kg     = 80  # ₹80 per kg
        money_wasted    = round(waste_kg * cost_per_kg, 2)

        records.append({
            "date":           row["date"],
            "day":            row["day"],
            "weather":        row["weather"],
            "total_students": row["total"],
            "food_prepared_kg": total_food_kg,
            "waste_kg":       waste_kg,
            "waste_percent":  round(waste_pct * 100, 2),
            "money_wasted_rs": money_wasted
        })

    df = pd.DataFrame(records)
    df.to_csv("dataset/waste_data.csv", index=False)
    print(f"✅ waste_data.csv → {len(df)} rows")

# ─── RUN ALL ──────────────────────────────────────────
if __name__ == "__main__":
    print("🍽️  Generating VIT Smart Mess Dataset...\n")
    generate_footfall()
    generate_feedback()
    generate_waste()
    print("\n🎉 All datasets ready inside /dataset folder!")