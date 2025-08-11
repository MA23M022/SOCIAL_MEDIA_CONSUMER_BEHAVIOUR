import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# Import the ml model
model = None
with open(r"C:\Users\soura\Desktop\Resume_Projects\SOCIAL_MEDIA_PROJECT\backend\model\best_model.pkl", "rb") as f:
    model = pickle.load(f)


if model:
    print(f"Model exists")
else:
    print(f"Does not able to fetch out model")


print("-------------------------------Read dataset and Extract relevent columns from dataset -------------------------------")
# Load the dataset
raw_df = pd.read_csv(r'C:\Users\soura\Desktop\Resume_Projects\SOCIAL_MEDIA_PROJECT\data\preprocessed\cleaned_data.csv')
removed_outliers_data = raw_df.copy()



print("---------------------------------------------- Train test split -------------------------------------------------------")

X = removed_outliers_data.drop(columns=["log_tw_data"])
y = removed_outliers_data["log_tw_data"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
print(f'X-Train data shape : {X_train.shape}, y-train data shape : {y_train.shape}' )
print(f'X-test data shape : {X_test.shape}, y-test data shape : {y_test.shape}')


# Predictions
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_exp = np.expm1(y_test)

# Metrics
rmse = root_mean_squared_error(y_test_exp, y_pred)
r2 = r2_score(y_test_exp, y_pred)
print(f"Best Model RMSE: {rmse:.4f}")
print(f"Best Model R²: {r2:.4f}")

