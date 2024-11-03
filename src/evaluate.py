# src/evaluate.py
import pandas as pd
import joblib
import json
import sys
import os

from utils.common import evaluateModel, readFolder

def evaluate(feature_file, model_input_file, metrics_file, target):
    df_features =  pd.read_csv(feature_file)
    df_test =  pd.read_csv("data/X_test.csv")

    features = df_features['feature'].values

    X_test = df_test.drop(columns=[target], errors='ignore')
    y_test = df_test[target]
    X_test = X_test[features]

    rootPath = os.getcwd()
    models_list = readFolder("models","pkl")

    metrics_output = {}
    for model_name in models_list:   
        print(model_name)
        #We need to concat only the filename because into the readfolder method we change our target path
        model_fullPath = os.path.join(os.getcwd(),model_name)
        model = joblib.load(model_fullPath)
        metrics = evaluateModel(model,X_test,y_test,params['train']['CV'])
        metrics_output[model_name] = metrics
    
    os.chdir(rootPath)   # Restore the base root path

    # Guardar métricas en un archivo JSON
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=4)

    print(f"Métricas guardadas en {metrics_file}")
   
    

if __name__ == "__main__":
    feature_input_file = sys.argv[1]
    #model_folder = sys.argv[2]
    metrics_file = sys.argv[2]
    params_file = sys.argv[3]

    import yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    target = params['preprocessing']['target']

    evaluate(feature_input_file, "", metrics_file, target)
