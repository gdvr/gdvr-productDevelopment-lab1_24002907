# src/preprocess.py
import pandas as pd
import sys

def preprocess(input_file, output_file, features, target):
    # Cargar el dataset
    df = pd.read_csv(input_file)

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Seleccionar solo las columnas necesarias
    columns = features + [target]
    df = df[columns]

    # Guardar el dataset limpio
    df.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Datos guardados en {output_file}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo de salida, características y objetivo
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    # Leer parámetros desde params.yaml
    import yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)
