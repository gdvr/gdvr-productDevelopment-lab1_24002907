# src/train.py
import pandas as pd
import joblib
import sys
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def train(input_file, model_file, params_file):
    # Cargar el dataset limpio
    df = pd.read_csv(input_file)

    # Leer los hiperparámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Separar características y variable objetivo
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X = df[features]
    y = df[target]

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )

    # Entrenar el modelo
    model = Ridge(alpha=params['train']['alpha'])
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(model, model_file)
    print(f"Modelo entrenado y guardado en {model_file}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo del modelo, archivo de hiperparámetros
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, model_file, params_file)
