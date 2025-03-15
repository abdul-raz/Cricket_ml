# import pandas as pd
# import numpy as np
# import os

# # 📌 Load datasets (Ensure correct file paths)
# deliveries_path = "E:/Projects/cricket_ml/dataset/deliveries.csv"
# matches_path = "E:/Projects/cricket_ml/dataset/matches.csv"

# deliveries = pd.read_csv(deliveries_path)
# matches = pd.read_csv(matches_path)

# print("\n✅ Datasets Loaded Successfully!")
# print(f"🔢 Deliveries: {deliveries.shape}, Matches: {matches.shape}")

# # 📌 Rename 'id' in matches.csv to 'match_id' for merging
# matches.rename(columns={"id": "match_id"}, inplace=True)

# # 📌 Merge deliveries with matches on 'match_id'
# df = deliveries.merge(matches, on="match_id", how="left")

# # ✅ Select necessary columns
# df = df[[
#     "match_id", "inning", "batting_team", "bowling_team", "over", "ball", "batter",
#     "batsman_runs", "venue", "season"
# ]]

# # ✅ Compute Total Runs & Balls Faced Per Player Per Match
# df["total_balls_faced"] = df.groupby(["match_id", "batter"])["ball"].transform("count")
# df["total_runs_scored"] = df.groupby(["match_id", "batter"])["batsman_runs"].transform("sum")

# # 📌 Categorize Overs (Powerplay, Middle, Death)
# def categorize_over(over):
#     if over <= 6:
#         return "Powerplay"
#     elif over <= 15:
#         return "Middle"
#     else:
#         return "Death"

# df["over_category"] = df["over"].apply(categorize_over)

# # ✅ Calculate Strike Rate for Different Over Phases
# strike_rates = df.groupby(["batter", "over_category"]).agg(
#     runs_scored=("batsman_runs", "sum"),
#     balls_faced=("ball", "count")
# ).reset_index()

# strike_rates["strike_rate"] = (strike_rates["runs_scored"] / strike_rates["balls_faced"]) * 100
# strike_rates.fillna(0, inplace=True)

# # ✅ Compute Player's Average Performance at Venues
# venue_avg_runs = df.groupby(["batter", "venue"]).agg(
#     venue_runs=("batsman_runs", "sum"),
#     venue_matches=("match_id", "nunique")
# ).reset_index()

# venue_avg_runs["venue_avg_runs"] = venue_avg_runs["venue_runs"] / venue_avg_runs["venue_matches"]
# venue_avg_runs.drop(columns=["venue_runs", "venue_matches"], inplace=True)

# # 📌 Aggregate Player's Overall Stats
# player_stats = df.groupby(["batter", "batting_team", "bowling_team", "venue", "season", "over_category"]).agg(
#     total_runs=("batsman_runs", "sum"),
#     balls_faced=("ball", "count")
# ).reset_index()

# # ✅ Merge Strike Rate & Venue Performance
# player_stats = player_stats.merge(strike_rates, on=["batter", "over_category"], how="left")
# player_stats = player_stats.merge(venue_avg_runs, on=["batter", "venue"], how="left")

# # ✅ Handle Missing Values
# player_stats.fillna(0, inplace=True)

# # 📌 Save Preprocessed Dataset
# output_path = "E:/Projects/cricket_ml/dataset/preprocessed_ipl_data.csv"
# player_stats.to_csv(output_path, index=False)

# print("\n✅ Preprocessing Completed! Data saved at:", output_path)
# print(f"📊 Final Dataset Shape: {player_stats.shape}")
# print(f"🔑 Columns: {player_stats.columns.tolist()}")
import pandas as pd
import os

# 📌 Load Preprocessed Dataset
input_path = "E:/Projects/cricket_ml/dataset/preprocessed_ipl_data.csv"
output_path = "E:/Projects/cricket_ml/dataset/final_preprocessed_ipl_data.csv"

df = pd.read_csv(input_path)

# ✅ Remove Redundant Columns
df.drop(columns=["balls_faced_x", "balls_faced_y", "runs_scored"], inplace=True)

# ✅ Save Cleaned Dataset
df.to_csv(output_path, index=False)

print("\n✅ Data Cleaning Completed! Final Dataset Saved at:", output_path)
print(f"📊 Final Shape: {df.shape}")
print(f"🔑 Columns: {df.columns.tolist()}")
