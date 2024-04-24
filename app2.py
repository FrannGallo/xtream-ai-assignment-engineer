from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from io import StringIO
import joblib
from sklearn.preprocessing import TargetEncoder, StandardScaler
import json
import numpy as np
import pandas as pd
from BankNotes import BankNote
import uvicorn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    
    # Supongamos que tienes un DataFrame df con las variables categóricas
    categorical_columns = ['cut', 'color', 'clarity']
    numeric_columns = ["depth", "table", "carat",'x', 'y', 'z']
  
    # Crea un nuevo DataFrame solo con las variables numéricas
    numeric_data = data.loc[:, numeric_columns]
  
    # # NUMERIC SCALAR
    global scaler
    scaler = StandardScaler().fit(numeric_data)
    scaled_numeric_data = scaler.transform(numeric_data)
  
    # Crea un DataFrame con las variables numéricas escaladas y las variables categóricas sin cambios
    df_scaled = pd.DataFrame(scaled_numeric_data, columns=numeric_columns)
    for column in categorical_columns:
        df_scaled[column] = data[column]
    # Reordena las columnas para que el DataFrame resultante tenga el mismo orden que el original
    df_scaled = df_scaled[data.columns]
  
    # Crea una instancia de LabelEncoder
    encoder = LabelEncoder()
    df_encoded= df_scaled.copy()
  
    # Aplica el LabelEncoder a cada columna categórica
    for col in categorical_columns:
        df_encoded[col] = encoder.fit_transform(data[col])
  
    # Ahora las variables categóricas han sido codificadas con números enteros
    print (df_encoded)
    return df_encoded
def preprocess_newdata(data):
    
    # Supongamos que tienes un DataFrame df con las variables categóricas
    categorical_columns = ['cut', 'color', 'clarity']
    numeric_columns = ["depth", "table", "carat",'x', 'y', 'z']
  
    # Crea un nuevo DataFrame solo con las variables numéricas
    numeric_data = data.loc[:, numeric_columns]
    # # NUMERIC SCALAR
    scaled_numeric_data = scaler.transform(numeric_data)
  
    # Crea un DataFrame con las variables numéricas escaladas y las variables categóricas sin cambios
    df_scaled = pd.DataFrame(scaled_numeric_data, columns=numeric_columns)
    for column in categorical_columns:
        df_scaled[column] = data[column]
    # Reordena las columnas para que el DataFrame resultante tenga el mismo orden que el original
    df_scaled = df_scaled[data.columns]
  

    # Ahora las variables categóricas han sido codificadas con números enteros
    print (df_scaled)
    return df_scaled

diamonds = pd.read_csv('C:/Users/fran_/Desktop/Embryioxite/Challenge/diamonds.csv')
app = FastAPI()

model = joblib.load('Modelo (1)')


def process_data(df):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    print(df)
    df = pd.DataFrame(df)
    print(df)
    
    diamonds_x= diamonds.drop(columns= "price")
    diamonds_y = pd.DataFrame(diamonds.price)
    df_x  = df
    print(df_x)
    

    #
    diamonds_x = preprocess_data(diamonds_x)
    print(diamonds_x)
    df_x =  preprocess_newdata(df_x)
    print(df_x)
    #
    predictions = model.predict(df_x)
    # Reentrenar el modelo con las predicciones
    diamonds_x = np.concatenate((diamonds_x, df_x), axis=0)  # Agregar las características de prueba a las de entrenamiento
    diamonds_y = np.concatenate((diamonds_y, pd.DataFrame(predictions)), axis=0)
    # Reentrenar el modelo con los datos actualizados
    model.fit(diamonds_x, diamonds_y)

    # Devolver el modelo reentrenado y las predicciones
    return model, predictions , diamonds_x, diamonds_y

""" data= {
    "carat": [0.90],
    "cut": [2],
    "color": [3],
    "clarity": [3],
    "depth": [61.7],
    "table": [57.0],
    "x": [6.17],
    "y": [6.21],
    "z": [4.11]
}
prediction = model.predict(data)
model, int_prices, diamonds_x, diamonds_y = process_data(data)
print(data)
data = pd.DataFrame(data)
data['carat']
"""

@app.post('/predict_one')
async def predict_banknote(data: BankNote):
    # Convierte los datos en un DataFrame de pandas
    data  = pd.json_normalize(data)
    # Procesa los datos y hace predicciones
    model, prediction, diamonds_x, diamonds_y = process_data(data)
    # Imprime el resultado de la predicción
    print(f"price is: {prediction}")
    # Clasifica la predicción en "PRICELESS" o "Not to shinny" según un umbral
    if prediction[0] > 7500:
        prediction_ = "PRICELESS"
    else:
        prediction_ = "Not to shinny"
    # Devuelve la predicción y el precio predicho como un diccionario
    return {
        'prediction': prediction_,
        'Price_Prediction': prediction[0]
        }


"""data= {
    "carat": 1.20,
    "cut": 4,
    "color": 5,
    "clarity": 1,
    "depth": 61.1,
    "table": 58.0,
    "x": 6.88,
    "y": 6.80,
    "z": 4.18
  }
"""
"""
@app.post('/predict_many')
async def predict_many(json_data: FeaturesList):
    df = pd.DataFrame(json_data.to_list())
    null_ids = df['id'].isnull().sum() > 0
    if null_ids:
        return {'error': 'IDs are mandatory. Please add them as a field in your JSON'}
    df.set_index('id', inplace=True)
    int_prices = process_data(df)
    results = dict(zip(df.index, int_prices))
    return {'results': results}

@app.post('/predict_many_csv')
async def predict_many_csv(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        return {"error": "Input must be a csv file"}
    content = await file.read()
    content = content.decode()
    df = pd.read_csv(StringIO(content))
    features = {'id', 'x', 'y', 'z', 'color', 'clarity'}
    if set(df.columns) != features:
        return {'error': f"These features are missing: {features-set(df.columns)}"}
    null_values = df.isnull().sum().sum()
    if null_values > 0:
        return {'error': f"There are {null_values} null values in the file. Please check them"}
    prices = process_data(df.copy())
    df['price'] = prices
    stream = StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=prices.csv'
    return response
"""
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app_copia:app --reload