from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load ML Model & Preprocessing Tools
model = joblib.load("model/batsman_performance_model.pkl")
model_features = joblib.load("model/features.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return jsonify({"message": "🏏 FantasyEdge IPL Prediction API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Parse JSON Request
        data = request.get_json()

        # ✅ Extract Input Values
        batsman_name = data.get("batsman_name", "").strip()
        test_venue = data.get("venue", "").strip()
        batting_team = data.get("batting_team", "").strip()
        bowling_team = data.get("bowling_team", "").strip()
        over_category = data.get("over_category", "").strip()

        # ✅ Load Preprocessed Data
        df = pd.read_csv("dataset/final_preprocessed_ipl_data.csv")
        df["batter"] = df["batter"].str.strip()

        # ✅ Check if Batsman Exists in Data
        batsman_stats = df[df["batter"] == batsman_name]
        if batsman_stats.empty:
            return jsonify({"error": f"No data found for '{batsman_name}'. Check spelling or dataset."})

        # ✅ Ensure player has enough data (at least 10 innings)
        if batsman_stats.shape[0] < 10:
            return jsonify({"error": f"Insufficient data for '{batsman_name}'. Likely a bowler or part-time batsman!"})

        # ✅ Get Latest Stats of Batsman
        latest_stats = batsman_stats.sort_values(by="season", ascending=False).iloc[0]

        # ✅ Compute 'batting_vs_opponent' if missing
        if "batting_vs_opponent" not in latest_stats:
            latest_stats["batting_vs_opponent"] = (latest_stats["venue_avg_runs"] * 0.6) + (latest_stats["strike_rate"] * 0.4)

        # ✅ Prepare Test Data (Numerical Features)
        test_data = pd.DataFrame([{
            "strike_rate": latest_stats["strike_rate"],
            "venue_avg_runs": latest_stats["venue_avg_runs"],
            "batting_vs_opponent": latest_stats["batting_vs_opponent"]
        }])

        # ✅ Ensure Correct Feature Order Before Scaling
        test_data = test_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # ✅ Apply Scaling
        test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=scaler.feature_names_in_)

        # ✅ Encode Categorical Features (Venue, Teams, Over Category)
        categorical_features = {col: 0 for col in model_features if col.startswith(("venue_", "batting_team_", "bowling_team_", "over_category_", "season_"))}
        categorical_features[f"venue_{test_venue}"] = 1 if f"venue_{test_venue}" in categorical_features else 0
        categorical_features[f"batting_team_{batting_team}"] = 1 if f"batting_team_{batting_team}" in categorical_features else 0
        categorical_features[f"bowling_team_{bowling_team}"] = 1 if f"bowling_team_{bowling_team}" in categorical_features else 0
        categorical_features[f"over_category_{over_category}"] = 1 if f"over_category_{over_category}" in categorical_features else 0

        # ✅ Encode Latest Season
        latest_season = f"season_{latest_stats['season']}"
        if latest_season in categorical_features:
            categorical_features[latest_season] = 1

        # ✅ Merge Categorical Features with Scaled Test Data
        categorical_df = pd.DataFrame([categorical_features])

        # ✅ **Fix Duplicate Columns Issue**
        test_data_scaled = test_data_scaled.loc[:, ~test_data_scaled.columns.duplicated()].copy()
        categorical_df = categorical_df.loc[:, ~categorical_df.columns.duplicated()].copy()

        # ✅ **Concatenate Features**
        test_data_final = pd.concat([test_data_scaled, categorical_df], axis=1)

        # ✅ **Fix Duplicate Column Labels Before Reindexing**
        test_data_final = test_data_final.loc[:, ~test_data_final.columns.duplicated()].copy()

        # ✅ **Ensure All Features from Model Are Present**
        test_data_final = test_data_final.reindex(columns=model_features, fill_value=0)

        # ✅ Debugging: Print Encoded Test Data
        print("\n✅ Encoded Test Data (First Row):")
        print(test_data_final.iloc[0])

        # ✅ Make Prediction
        predicted_runs = model.predict(test_data_final)[0]

        # ✅ Return Response
        return jsonify({
            "batsman_name": batsman_name,
            "venue": test_venue,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "over_category": over_category,
            "predicted_runs": round(predicted_runs, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
