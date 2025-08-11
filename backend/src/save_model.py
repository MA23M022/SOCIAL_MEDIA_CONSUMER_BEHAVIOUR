import pickle
import pandas as pd

# Import the ml model
model = None
with open(r"C:\Users\soura\Desktop\Resume_Projects\SOCIAL_MEDIA_PROJECT\mlartifacts\865518813492507599\models\m-77ebe61ab5354d8d8470c969f91b95cb\artifacts\model.pkl", "rb") as f:
    model = pickle.load(f)
    with open(r'C:\Users\soura\Desktop\Resume_Projects\SOCIAL_MEDIA_PROJECT\backend\model\best_model.pkl', 'wb') as file:
        pickle.dump(model, file)



if model:
    print(f"Model has been saved")
else:
    print(f"Does not able to fetch out model")
