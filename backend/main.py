from fastapi import FastAPI
from fastapi.responses import JSONResponse
from backend.schema.user_input import UserInput
from backend.model.predict import model, MODEL_VERSION, predict_output
from backend.schema.model_output import PredictionResponse

# create end point APIs :-
app = FastAPI()


@app.get("/")
def welcome():
    return {"message" : "Welcome to Social Media Consumer Activity Prediction App"}


@app.get("/health")
def health_check():
    return {
            "status" : "OK",
            "model_version" : MODEL_VERSION,
            "model_loaded" : model is not None
        }

@app.post("/predict", response_model = PredictionResponse)
def predict_activity(data : UserInput):
    if data.Agency == "Unknown":
        return JSONResponse(status_code=404 , content = "The given agency has not been seen earlier")
    
    input_data = {
            "Agency" : data.Agency,
            "Month_Sampled" : data.Month_Sampled,
            "log_fb_data" : data.log_fb_data
        }
    
    try:
        model_pred = predict_output(input_data)
        return JSONResponse(status_code = 200, content = {"twitter" : model_pred["twitter_prediction"]})
    except Exception as e:
        return JSONResponse(status_code = 500, content = str(e))
    
