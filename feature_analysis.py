import matplotlib.pyplot as plt
import pandas as pd
import joblib

model = joblib.load("crz_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
ohe = preprocessor.named_transformers_['cat']
encoded_feature_names = ohe.get_feature_names_out(preprocessor.transformers_[0][2])
all_feature_names = list(encoded_feature_names) + ["Hour of Day"]

# Get importances
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][:15][::-1], importance_df["Importance"][:15][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 15 Features Influencing CRZ Entries")
plt.tight_layout()
plt.show()


