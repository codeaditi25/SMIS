import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── STYLE SETUP ────────────────────────────────────────
plt.rcParams["figure.facecolor"] = "#0f0f0f"
plt.rcParams["axes.facecolor"]   = "#1a1a2e"
plt.rcParams["axes.labelcolor"]  = "white"
plt.rcParams["xtick.color"]      = "white"
plt.rcParams["ytick.color"]      = "white"
plt.rcParams["text.color"]       = "white"
plt.rcParams["grid.color"]       = "#2a2a4a"
plt.rcParams["grid.linestyle"]   = "--"
plt.rcParams["grid.alpha"]       = 0.5

COLORS = ["#e94560", "#0f3460", "#533483", "#06d6a0",
          "#ffd166", "#ef476f", "#118ab2", "#073b4c"]

# ── LOAD DATA ──────────────────────────────────────────
footfall = pd.read_csv("dataset/footfall_data.csv")
feedback = pd.read_csv("dataset/dish_feedback.csv")
waste    = pd.read_csv("dataset/waste_data.csv")

footfall["date"] = pd.to_datetime(footfall["date"])
waste["date"]    = pd.to_datetime(waste["date"])
feedback["date"] = pd.to_datetime(feedback["date"])

print("✅ Data loaded successfully!")
print(f"   Footfall : {len(footfall)} records")
print(f"   Feedback : {len(feedback)} records")
print(f"   Waste    : {len(waste)} records\n")

# ══════════════════════════════════════════════════════
# CHART 1 — Daily Footfall Trend
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(footfall["date"], footfall["breakfast"],
        color="#ffd166", linewidth=1.5, label="Breakfast", alpha=0.9)
ax.plot(footfall["date"], footfall["lunch"],
        color="#06d6a0", linewidth=1.5, label="Lunch", alpha=0.9)
ax.plot(footfall["date"], footfall["dinner"],
        color="#e94560", linewidth=1.5, label="Dinner", alpha=0.9)
ax.fill_between(footfall["date"], footfall["lunch"],
                alpha=0.1, color="#06d6a0")
ax.set_title("📈 Daily Student Footfall Trend — VIT Bhopal Mess",
             fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig("dataset/chart1_footfall_trend.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Chart 1 saved!")

# ══════════════════════════════════════════════════════
# CHART 2 — Average Footfall by Day of Week
# ══════════════════════════════════════════════════════
day_order  = ["Monday","Tuesday","Wednesday",
              "Thursday","Friday","Saturday","Sunday"]
day_avg    = footfall.groupby("day")["total"].mean().reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(day_avg.index, day_avg.values,
              color=COLORS[:7], edgecolor="white",
              linewidth=0.5, width=0.6)
for bar, val in zip(bars, day_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 20,
            f"{int(val)}", ha="center", va="bottom",
            fontsize=10, color="white", fontweight="bold")
ax.set_title("📅 Average Student Footfall by Day of Week",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Day", fontsize=12)
ax.set_ylabel("Avg Students", fontsize=12)
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig("dataset/chart2_day_footfall.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Chart 2 saved!")

# ══════════════════════════════════════════════════════
# CHART 3 — Sentiment Distribution (Pie Chart)
# ══════════════════════════════════════════════════════
sentiment_counts = feedback["sentiment"].value_counts()
explode = (0.05, 0.05, 0.05)

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    colors=["#06d6a0", "#ffd166", "#e94560"],
    explode=explode,
    startangle=140,
    textprops={"color": "white", "fontsize": 13},
    wedgeprops={"edgecolor": "#0f0f0f", "linewidth": 2}
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
ax.set_title("😊 Overall Meal Sentiment Distribution",
             fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("dataset/chart3_sentiment_pie.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Chart 3 saved!")

# ══════════════════════════════════════════════════════
# CHART 4 — Top 5 Loved & Hated Dishes
# ══════════════════════════════════════════════════════
dish_sentiment = feedback.groupby(
    ["dish","sentiment"])["votes"].sum().unstack(fill_value=0)

if "Positive" in dish_sentiment.columns:
    top_loved  = dish_sentiment["Positive"].nlargest(5)
if "Negative" in dish_sentiment.columns:
    top_hated  = dish_sentiment["Negative"].nlargest(5)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# loved
axes[0].barh(top_loved.index, top_loved.values,
             color="#06d6a0", edgecolor="white", linewidth=0.5)
axes[0].set_title("🏆 Top 5 Most Loved Dishes",
                  fontsize=14, fontweight="bold")
axes[0].set_xlabel("Positive Votes", fontsize=11)
axes[0].grid(True, axis="x")
for i, v in enumerate(top_loved.values):
    axes[0].text(v + 5, i, str(int(v)),
                 va="center", fontsize=10, color="white")

# hated
axes[1].barh(top_hated.index, top_hated.values,
             color="#e94560", edgecolor="white", linewidth=0.5)
axes[1].set_title("👎 Top 5 Most Disliked Dishes",
                  fontsize=14, fontweight="bold")
axes[1].set_xlabel("Negative Votes", fontsize=11)
axes[1].grid(True, axis="x")
for i, v in enumerate(top_hated.values):
    axes[1].text(v + 5, i, str(int(v)),
                 va="center", fontsize=10, color="white")

plt.suptitle("🍽️ Dish Popularity Analysis — VIT Bhopal Mess",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("dataset/chart4_dish_popularity.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Chart 4 saved!")

# ══════════════════════════════════════════════════════
# CHART 5 — Monthly Food Waste Trend
# ══════════════════════════════════════════════════════
waste["month"] = waste["date"].dt.to_period("M")
monthly_waste  = waste.groupby("month")["waste_kg"].sum()
monthly_money  = waste.groupby("month")["money_wasted_rs"].sum()

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.bar([str(m) for m in monthly_waste.index],
        monthly_waste.values,
        color="#533483", alpha=0.8,
        edgecolor="white", linewidth=0.5,
        label="Waste (kg)")
ax2.plot([str(m) for m in monthly_money.index],
         monthly_money.values,
         color="#ffd166", linewidth=2.5,
         marker="o", markersize=6,
         label="Money Wasted (₹)")

ax1.set_title("🗑️ Monthly Food Waste & Money Lost — VIT Bhopal Mess",
              fontsize=15, fontweight="bold", pad=15)
ax1.set_xlabel("Month", fontsize=12)
ax1.set_ylabel("Food Wasted (kg)", fontsize=12, color="#533483")
ax2.set_ylabel("Money Wasted (₹)", fontsize=12, color="#ffd166")
ax1.tick_params(axis="x", rotation=45)

lines1 = mpatches.Patch(color="#533483", label="Waste kg")
lines2 = mpatches.Patch(color="#ffd166", label="Money ₹")
ax1.legend(handles=[lines1, lines2], fontsize=11)
ax1.grid(True, axis="y")
plt.tight_layout()
plt.savefig("dataset/chart5_waste_trend.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Chart 5 saved!")

# ══════════════════════════════════════════════════════
# CHART 6 — Heatmap: Day vs Weather vs Footfall
# ══════════════════════════════════════════════════════
heat_data = footfall.groupby(
    ["day","weather"])["total"].mean().unstack()
heat_data = heat_data.reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(heat_data,
            annot=True, fmt=".0f",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="#0f0f0f",
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"})
ax.set_title("🌡️ Footfall Heatmap: Day vs Weather",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Weather", fontsize=12)
ax.set_ylabel("Day of Week", fontsize=12)
plt.tight_layout()
plt.savefig("dataset/chart6_heatmap.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Chart 6 saved!")

# ══════════════════════════════════════════════════════
# SUMMARY STATS
# ══════════════════════════════════════════════════════
print("\n" + "═"*50)
print("📊  VIT SMART MESS — KEY INSIGHTS")
print("═"*50)
print(f"🧑 Avg daily students     : {int(footfall['total'].mean())}")
print(f"📅 Busiest day            : {day_avg.idxmax()}")
print(f"📅 Quietest day           : {day_avg.idxmin()}")
print(f"😊 Positive feedback      : {sentiment_counts.get('Positive',0)} reviews")
print(f"😞 Negative feedback      : {sentiment_counts.get('Negative',0)} reviews")
print(f"🗑️  Total food wasted      : {waste['waste_kg'].sum():.0f} kg")
print(f"💰 Total money wasted     : ₹{waste['money_wasted_rs'].sum():,.0f}")
print(f"💡 Avg waste per day      : {waste['waste_kg'].mean():.1f} kg")
print("═"*50)
print("\n✅ All 6 charts saved in /dataset folder!")