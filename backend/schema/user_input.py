from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Annotated
import numpy as np
from backend.config.agency import Agency_set



class UserInput(BaseModel):
    Agency : Annotated[str, Field(..., description = "Name of the Agency that shows the products", examples = ["City Store", "City Charter"])]
    Month_Sampled : Annotated[int, Field(..., gt = 0, lt = 13, description = "On which Month the product has been shown", examples = [1, 2])]
    fb_data : Annotated[float, Field(..., gt = 0, description = "Likes on the product at the facebbok platform", examples = [230, 5000])]

    @field_validator("Agency")
    @classmethod
    def check_agency(cls, v:str)->str:
        if v in Agency_set:
            return v
        else:
            return "Unknown"


    @computed_field
    @property
    def log_fb_data(self) -> float:
        return np.log1p(self.fb_data)
    

