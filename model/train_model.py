import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 📌 Ensure model directory exists
os.makedirs("model", exist_ok=True)

# 📌 **Load Correct Dataset**
df = pd.read_csv("dataset/final_preprocessed_ipl_data.csv")  # ✅ Fixed dataset name

print("\n✅ Dataset Loaded Successfully!")
print(f"🔢 Total Rows: {df.shape[0]} | 📊 Columns: {df.columns.tolist()}")

# 📌 Feature Engineering: Create meaningful interactions
df["strike_rate"] = df["strike_rate"] / 200  # Normalize strike rate (max ~200)

# 🏏 Weighted Batting Performance Against Opponent
df["batting_vs_opponent"] = (df["venue_avg_runs"] * 0.6) + (df["strike_rate"] * 0.4)

# 📌 Drop unnecessary columns
df.drop(columns=["batter"], inplace=True)  # Remove player name (batter)

# 📌 Define Features (X) and Target Variable (y)
X = df.drop(columns=["total_runs"])  # Features (Exclude target)
y = df["total_runs"]  # Target Variable

# 📌 Convert categorical variables into numerical (One-Hot Encoding)
X = pd.get_dummies(X)

# 📌 Normalize numerical features
scaler = MinMaxScaler()
num_cols = ["strike_rate", "season", "batting_vs_opponent", "venue_avg_runs"]

# ✅ Ensure all numerical columns exist before scaling
num_cols = [col for col in num_cols if col in X.columns]
X[num_cols] = scaler.fit_transform(X[num_cols])

# 📌 Save feature names for consistent encoding in predictions
feature_names = X.columns.tolist()
joblib.dump(feature_names, "model/features.pkl")
joblib.dump(scaler, "model/scaler.pkl")  # Save scaler for later use

print("\n🔠 One-Hot Encoding & Scaling Completed!")
print(f"🆕 Features Used for Training: {len(feature_names)}")

# 📌 Split the data into Training (80%) & Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Training Samples: {len(X_train)} | 🧪 Testing Samples: {len(X_test)}")

# 📌 Initialize and Train the Model (Tunable Hyperparameters)
model = RandomForestRegressor(
    n_estimators=300,  # Number of trees
    max_depth=12,  # Depth of trees
    min_samples_split=5,  # Minimum samples to split a node
    min_samples_leaf=2,  # Minimum samples per leaf node
    random_state=42
)
model.fit(X_train, y_train)

print("\n🚀 Model Training Completed!")

# 📌 Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 📌 Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance Metrics:")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# 📌 Save the trained model
joblib.dump(model, "model/batsman_performance_model.pkl")

print("\n✅ Model Successfully Saved at: 'model/batsman_performance_model.pkl'")
print("✅ Feature Names & Scaler Saved.")

# 📌 DEBUG: Print feature importances
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
print("\n📊 **Top 10 Most Important Features in Model:**")
print(feature_importance.sort_values(by="Importance", ascending=False).head(10))

# 📌 Save feature importance for further analysis
feature_importance.to_csv("model/feature_importance.csv", index=False)
print("\n✅ Feature Importance Saved at: 'model/feature_importance.csv'")
