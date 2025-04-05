import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load and prep data
df = pd.read_csv("data.csv")
features = [
    "Detection Group", "Detection Region", "Vehicle Class",
    "Day of Week", "Hour of Day", "Time Period"
]
target = "CRZ Entries"
df = df[features + [target]].dropna()
X = df[features]
y = df[target]

# Preprocessing
categorical_cols = X.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess data manually so we can reuse it
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Custom training loop with incremental trees
print("🌱 Training RandomForest tree by tree...\n")
n_estimators = 10
forest = RandomForestRegressor(
    n_estimators=1,
    warm_start=True,
    random_state=42
)

mae_list = []
r2_list = []

for i in range(1, n_estimators + 1):
    forest.set_params(n_estimators=i)
    forest.fit(X_train_processed, y_train)
    y_pred = forest.predict(X_test_processed)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae_list.append(mae)
    r2_list.append(r2)
    print(f"🌳 Tree {i:03d}: MAE = {mae:.2f} | R² = {r2:.4f}")

print("\n✅ Training complete!")
print(f"🎯 Final MAE: {mae_list[-1]:.2f}, Final R²: {r2_list[-1]:.4f}")

joblib.dump(forest, "crz_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")