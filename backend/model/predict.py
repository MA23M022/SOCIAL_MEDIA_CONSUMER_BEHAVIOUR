import pickle
import pandas as pd
import numpy as np

# Import the ml model
model = None
with open(r"C:/Users/soura/Desktop/Resume_Projects/SOCIAL_MEDIA_PROJECT/backend/model/best_model.pkl", "rb") as f:
    model = pickle.load(f)


MODEL_VERSION = "1.0.0"

def predict_output(input_data : dict) -> dict:
    input_df = pd.DataFrame([input_data])

    log_prediction = model.predict(input_df)[0]

    prediction = np.expm1(log_prediction)
    return {"twitter_prediction" : prediction}