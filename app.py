# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd
import joblib
# 2. Create the app object
app = FastAPI()
classifier = joblib.load('Modelo (1)')
#4493
# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
async def index():
    return {'message': 'Hello, World'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
print(classifier)
@app.post('/predict')
async def predict_banknote(data:BankNote):
    data = data.dict() 
    carat=  data['carat']
    cut=    data['cut']
    color=  data['color']
    clarity=    data['clarity'] 
    depth=  data['depth']
    table=  data['table']
    x=  data['x']
    y=  data['y']
    z=  data['z']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    #prediction = classifier.predict(LIST)
    prediction = classifier.predict([[carat,cut,color,clarity,depth,table,x,y,z]])
    #prediction = classifier.predict([[0.9,4,2,5,61.7,57,6.17,6.21,3.82]])
    print(f"price is: {prediction}")
    if(prediction[0]>7500):
        prediction_="PRICELESS"
    else:
        prediction_="Not to shinny"
    return {
        'prediction': prediction_,
        'Precio_Predicho': float(prediction[0])
}
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload