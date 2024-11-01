# src/preprocess.py
import pandas as pd
import sys
import yaml

from utils.common import categorizeColumns,  detectInvalidValues, handlingEmptyValues

def preprocess(input_file, output_file, features, target):
    df = pd.read_csv(input_file)
    df = df.dropna()
    columns = features + [target]
    df = df[columns]

    continuas, discretas, categoricas = categorizeColumns(df[features])
    detectInvalidValues(df[columns])
    handlingEmptyValues(df[columns].copy(),continuas + discretas)


    df.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Datos guardados en {output_file}")

    params['continuas'] = continuas
    params['discretas'] = discretas
    params['categoricas'] = categoricas

    with open("params.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    with open(params_file) as f:
        params = yaml.safe_load(f)
    
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    try:
        preprocess(input_file, output_file, features, target)
        sys.exit(0)  # Success
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        sys.exit(1)  # Error
