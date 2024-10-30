# src/preprocess.py
import pandas as pd
import sys
import yaml

from utils.common import categorizeColumns, detectInvalidValues, handlingEmptyValues, splitValuesForModel

def preprocess(input_file, output_file, features, target):
    df = pd.read_csv(input_file)
    df = df.dropna()
    columns = features + [target]
    df = df[columns]

    continuas, discretas, categoricas = categorizeColumns(df)
    detectInvalidValues( df[columns])
    handlingEmptyValues(df[columns].copy(),continuas + discretas)

    X = df[features]
    y = df[target]
    X_train, y_train,X_test,y_test,X_val, y_val = splitValuesForModel(X,y,params['train']['TEST_SIZE'],params['train']['VALIDATE_SIZE'],params['train']['RANDOM_STATE'])
    X_train[target] = y_train
    X_test[target] = y_test
    X_val[target] = y_val

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    X_val.to_csv("data/X_val.csv", index=False)

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
    
    print(params)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)
