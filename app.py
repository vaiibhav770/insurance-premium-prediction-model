from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.userinput import UserInput

import pickle
import pandas as pd

# import the ml model
with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)

MODEL_VERSION= '1.0.0'

app = FastAPI()


@app.get('/')
def home_page():
    return {'message':'go to /docs'}

@app.get('/health')
def health_check():
    return{
        'status':'OK',
        'VERSION' : MODEL_VERSION,
        'MODEL_LODED' : model is not None

    }

@app.post('/predict')
def predict_premium(data: UserInput):

    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={'predicted_category': prediction})





