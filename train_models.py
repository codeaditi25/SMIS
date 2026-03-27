import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── STYLE ──────────────────────────────────────────────
plt.rcParams["figure.facecolor"] = "#0f0f0f"
plt.rcParams["axes.facecolor"]   = "#1a1a2e"
plt.rcParams["axes.labelcolor"]  = "white"
plt.rcParams["xtick.color"]      = "white"
plt.rcParams["ytick.color"]      = "white"
plt.rcParams["text.color"]       = "white"
plt.rcParams["grid.color"]       = "#2a2a4a"
plt.rcParams["grid.alpha"]       = 0.5

print("🤖 Training VIT Smart Mess ML Models...\n")

# ══════════════════════════════════════════════════════
# MODEL 1 — FOOTFALL PREDICTOR
# ══════════════════════════════════════════════════════
print("═"*50)
print("📌 MODEL 1: Student Footfall Predictor")
print("═"*50)

footfall = pd.read_csv("dataset/footfall_data.csv")

# encode categorical columns
le_day     = LabelEncoder()
le_weather = LabelEncoder()
footfall["day_enc"]     = le_day.fit_transform(footfall["day"])
footfall["weather_enc"] = le_weather.fit_transform(footfall["weather"])

# features and target
X1 = footfall[["day_enc","weather_enc","exam_week","holiday"]]
y1 = footfall["total"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42)

# train
rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf1.fit(X1_train, y1_train)
y1_pred = rf1.predict(X1_test)

mae1 = mean_absolute_error(y1_test, y1_pred)
r2_1 = r2_score(y1_test, y1_pred)

print(f"✅ MAE  : {mae1:.2f} students")
print(f"✅ R²   : {r2_1:.4f}")
print(f"✅ Accuracy : {round(r2_1*100, 2)}%\n")

# feature importance chart
fig, ax = plt.subplots(figsize=(8, 5))
feat_names = ["Day","Weather","Exam Week","Holiday"]
importances = rf1.feature_importances_
bars = ax.barh(feat_names, importances,
               color=["#e94560","#06d6a0","#ffd166","#533483"],
               edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, importances):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=11,
            color="white", fontweight="bold")
ax.set_title("🔍 Feature Importance — Footfall Predictor",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Importance Score", fontsize=11)
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig("dataset/model1_feature_importance.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()

# actual vs predicted chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(len(y1_test[:60])),
        y1_test[:60].values,
        color="#06d6a0", linewidth=2,
        label="Actual", marker="o", markersize=4)
ax.plot(range(len(y1_pred[:60])),
        y1_pred[:60],
        color="#e94560", linewidth=2,
        label="Predicted", marker="s",
        markersize=4, linestyle="--")
ax.fill_between(range(len(y1_test[:60])),
                y1_test[:60].values,
                y1_pred[:60], alpha=0.1, color="#ffd166")
ax.set_title("📈 Actual vs Predicted Footfall",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Sample Index", fontsize=11)
ax.set_ylabel("Number of Students", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig("dataset/model1_actual_vs_predicted.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Model 1 charts saved!\n")

# ══════════════════════════════════════════════════════
# MODEL 2 — FOOD WASTE PREDICTOR
# ══════════════════════════════════════════════════════
print("═"*50)
print("📌 MODEL 2: Food Waste Predictor")
print("═"*50)

waste = pd.read_csv("dataset/waste_data.csv")

le_day2     = LabelEncoder()
le_weather2 = LabelEncoder()
waste["day_enc"]     = le_day2.fit_transform(waste["day"])
waste["weather_enc"] = le_weather2.fit_transform(waste["weather"])

X2 = waste[["day_enc","weather_enc",
            "total_students","food_prepared_kg"]]
y2 = waste["waste_kg"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42)

rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2.fit(X2_train, y2_train)
y2_pred = rf2.predict(X2_test)

mae2 = mean_absolute_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print(f"✅ MAE  : {mae2:.2f} kg")
print(f"✅ R²   : {r2_2:.4f}")
print(f"✅ Accuracy : {round(r2_2*100, 2)}%\n")

# actual vs predicted waste
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(len(y2_test[:60])),
        y2_test[:60].values,
        color="#ffd166", linewidth=2,
        label="Actual Waste", marker="o", markersize=4)
ax.plot(range(len(y2_pred[:60])),
        y2_pred[:60],
        color="#e94560", linewidth=2,
        label="Predicted Waste", marker="s",
        markersize=4, linestyle="--")
ax.set_title("🗑️ Actual vs Predicted Food Waste (kg)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Sample Index", fontsize=11)
ax.set_ylabel("Waste (kg)", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig("dataset/model2_waste_prediction.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()

# feature importance
fig, ax = plt.subplots(figsize=(8, 5))
feat_names2  = ["Day","Weather","Total Students","Food Prepared"]
importances2 = rf2.feature_importances_
bars2 = ax.barh(feat_names2, importances2,
                color=["#e94560","#06d6a0","#ffd166","#533483"],
                edgecolor="white", linewidth=0.5)
for bar, val in zip(bars2, importances2):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=11,
            color="white", fontweight="bold")
ax.set_title("🔍 Feature Importance — Waste Predictor",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Importance Score", fontsize=11)
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig("dataset/model2_feature_importance.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Model 2 charts saved!\n")

# ══════════════════════════════════════════════════════
# MODEL 3 — SENTIMENT CLASSIFIER
# ══════════════════════════════════════════════════════
print("═"*50)
print("📌 MODEL 3: Meal Sentiment Classifier")
print("═"*50)

feedback = pd.read_csv("dataset/dish_feedback.csv")

le_dish = LabelEncoder()
le_sent = LabelEncoder()
feedback["dish_enc"]      = le_dish.fit_transform(feedback["dish"])
feedback["sentiment_enc"] = le_sent.fit_transform(feedback["sentiment"])

X3 = feedback[["dish_enc","rating","votes"]]
y3 = feedback["sentiment_enc"]

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42)

rf3 = RandomForestClassifier(n_estimators=100, random_state=42)
rf3.fit(X3_train, y3_train)
y3_pred = rf3.predict(X3_test)

acc3 = accuracy_score(y3_test, y3_pred)
print(f"✅ Accuracy : {round(acc3*100, 2)}%\n")
print("Classification Report:")
print(classification_report(
    y3_test, y3_pred,
    target_names=le_sent.classes_))

# confusion matrix
cm = confusion_matrix(y3_test, y3_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            cmap="YlOrRd",
            xticklabels=le_sent.classes_,
            yticklabels=le_sent.classes_,
            linewidths=0.5,
            linecolor="#0f0f0f",
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("😊 Confusion Matrix — Sentiment Classifier",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig("dataset/model3_confusion_matrix.png",
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Model 3 charts saved!\n")

# ══════════════════════════════════════════════════════
# SAVE ALL MODELS
# ══════════════════════════════════════════════════════
print("═"*50)
print("💾 Saving all models...")
print("═"*50)

joblib.dump(rf1, "models/footfall_model.pkl")
joblib.dump(rf2, "models/waste_model.pkl")
joblib.dump(rf3, "models/sentiment_model.pkl")
joblib.dump(le_day,     "models/le_day.pkl")
joblib.dump(le_weather, "models/le_weather.pkl")
joblib.dump(le_dish,    "models/le_dish.pkl")
joblib.dump(le_sent,    "models/le_sent.pkl")

print("✅ footfall_model.pkl  saved!")
print("✅ waste_model.pkl     saved!")
print("✅ sentiment_model.pkl saved!")

# ══════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "═"*50)
print("🏆  MODEL PERFORMANCE SUMMARY")
print("═"*50)
print(f"🧑 Footfall Predictor  R²  : {round(r2_1*100,2)}%")
print(f"🗑️  Waste Predictor     R²  : {round(r2_2*100,2)}%")
print(f"😊 Sentiment Classifier Acc: {round(acc3*100,2)}%")
print("═"*50)
print("\n🎉 All 3 models trained and saved successfully!")