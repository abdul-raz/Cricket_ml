import pandas as pd
import joblib
import numpy as np

# 🔹 Load trained model
model = joblib.load("model/batsman_performance_model.pkl")

# 🔹 Load saved feature names & scaler
model_features = joblib.load("model/features.pkl")
scaler = joblib.load("model/scaler.pkl")

# 🔹 Load preprocessed IPL data
df = pd.read_csv("dataset/final_preprocessed_ipl_data.csv")  # Ensure correct dataset

# 🔹 Normalize player names
df["batter"] = df["batter"].str.strip()

# 🔹 Hardcoded Inputs
batsman_name = "WP Saha"
test_venue = "Narendra Modi Stadium"
batting_team = "Royal Challengers Bangalore"
bowling_team = "Kolkata Knight Riders"
over_category = "Middle"

print(f"\n🚀 Predicting for: {batsman_name} | Venue: {test_venue} | Batting Team: {batting_team} | Opponent: {bowling_team} | Over: {over_category}")

# 🔹 Filter data for the specific batsman
batsman_stats = df[df["batter"] == batsman_name]

# if batsman_stats.empty:
#     print(f"⚠️ No data found for '{batsman_name}'. Please check spelling or dataset.")
#     exit()
# 🔹 Ensure player has at least 10 innings in dataset
if batsman_stats.shape[0] < 10:
    print(f"⚠️ Insufficient batting data for '{batsman_name}'. Likely a bowler or a part-time batsman!")
    exit()
# 🔹 Use latest available stats for the player
latest_stats = batsman_stats.sort_values(by="season", ascending=False).iloc[0]

# ✅ **Check & Compute 'batting_vs_opponent' if missing**
if "batting_vs_opponent" not in df.columns:
    latest_stats["batting_vs_opponent"] = (latest_stats["venue_avg_runs"] * 0.6) + (latest_stats["strike_rate"] * 0.4)

# 🔹 Prepare test input data
test_data = pd.DataFrame([{
    "strike_rate": latest_stats["strike_rate"],
    "venue_avg_runs": latest_stats["venue_avg_runs"],
    "batting_vs_opponent": latest_stats["batting_vs_opponent"]
}])

# 🔹 Ensure test_data matches scaler's feature order
test_data = test_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# 🔹 Apply feature scaling
test_data[scaler.feature_names_in_] = scaler.transform(test_data[scaler.feature_names_in_])

# 🔹 Encode categorical features (venue, teams, overs)
categorical_features = {col: 0 for col in model_features if col.startswith(("venue_", "batting_team_", "bowling_team_", "over_category_"))}

# 🔹 Set correct venue, batting team, bowling team, and over category
categorical_features[f"venue_{test_venue}"] = 1 if f"venue_{test_venue}" in categorical_features else 0
categorical_features[f"batting_team_{batting_team}"] = 1 if f"batting_team_{batting_team}" in categorical_features else 0
categorical_features[f"bowling_team_{bowling_team}"] = 1 if f"bowling_team_{bowling_team}" in categorical_features else 0
categorical_features[f"over_category_{over_category}"] = 1 if f"over_category_{over_category}" in categorical_features else 0

# 🔹 Add categorical features to test_data
test_data = test_data.assign(**categorical_features)

# 🔹 Ensure test_data matches trained model features
test_data = test_data.reindex(columns=model_features, fill_value=0)

# 🔹 Debugging - Check processed test data
print("\n✅ Encoded Test Data (First Row):")
print(test_data.iloc[0])

# 🔹 Make prediction
predicted_runs = model.predict(test_data)[0]

# 🔹 Print the predicted runs
print(f"\n🎯 Predicted Runs for {batsman_name} at {test_venue} against {bowling_team}: {predicted_runs:.2f}")
