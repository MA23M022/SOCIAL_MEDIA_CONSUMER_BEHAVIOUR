# --------------------------------------- Import libraries  ------------------------------------------------
print("-------------------------------Import essential libraries --------------------------------------------")
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse



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



print("--------------------------------------- Create the preprocessor ----------------------------------------------------------")

categorical_features = ["Agency", "Month_Sampled"]
numerical_features = ["log_fb_data"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("num", "passthrough", numerical_features)
])


# -------------------------------------- MLflow Setup -------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Social Media Consumer Activity Prediction")

# -------------------------------------- Helper Function ----------------------------------------------
def train_and_log_model(model_name, estimator, param_grid, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        print(f"\n================ Training {model_name} ================\n")

        # Build pipeline
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("regressor", estimator)
        ])

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"Best params for {model_name}: {grid_search.best_params_}")

        # Predictions
        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_exp = np.expm1(y_test)

        # Metrics
        rmse = root_mean_squared_error(y_test_exp, y_pred)
        r2 = r2_score(y_test_exp, y_pred)
        print(f"{model_name} RMSE: {rmse:.4f}")
        print(f"{model_name} R²: {r2:.4f}")

        # Log params & metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Signature
        signature = infer_signature(X_train, best_model.predict(X_train))

        # Save model
        tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme
        artifact_path = model_name.replace(" ", "_")
        if tracking_scheme != "file":
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=artifact_path,
                registered_model_name=model_name.replace(" ", "_"),
                signature=signature
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=artifact_path,
                signature=signature
            )

        print(f"{model_name} saved to MLflow.")

# -------------------------------------- Model Param Grids --------------------------------------------
dt_param_grid = {
    'regressor__max_depth': [3, 5, 7, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

rf_param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

xgb_param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1, 0.3],
    'regressor__subsample': [0.7, 0.8, 0.9],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9],
    'regressor__gamma': [0, 0.1, 0.2]
}

# -------------------------------------- Run All Models -----------------------------------------------
train_and_log_model("Decision Tree Model", DecisionTreeRegressor(random_state=0), dt_param_grid, X_train, y_train, X_test, y_test)
train_and_log_model("Random Forest Model", RandomForestRegressor(random_state=42), rf_param_grid, X_train, y_train, X_test, y_test)
train_and_log_model("XGBoost Model", XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1), xgb_param_grid, X_train, y_train, X_test, y_test)

print("\n✅ All models trained, tuned, and logged to MLflow successfully!")

