# src/evaluate.py
import pandas as pd
import joblib
import json
import sys
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(input_file, model_file, metrics_file, params_file):
    # Cargar el dataset limpio
    df = pd.read_csv(input_file)

    # Leer los parámetros
    import yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Separar características y variable objetivo
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X = df[features]
    y = df[target]

    # Cargar el modelo entrenado
    model = joblib.load(model_file)

    # Realizar predicciones
    predictions = model.predict(X)

    # Calcular métricas
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    # Guardar métricas en un archivo JSON
    with open(metrics_file, 'w') as f:
        json.dump({'mse': mse, 'r2': r2}, f, indent=4)
    print(f"Métricas guardadas en {metrics_file}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo del modelo, archivo de métricas, archivo de parámetros
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    metrics_file = sys.argv[3]
    params_file = sys.argv[4]

    evaluate(input_file, model_file, metrics_file, params_file)
