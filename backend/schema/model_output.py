from pydantic import BaseModel, Field
from typing import Annotated


class PredictionResponse(BaseModel):
    twitter_prediction : Annotated[float, Field(..., description = "Predicted activity on twitter", examples = [367])]