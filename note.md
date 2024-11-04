## Tabla de contenido

### Comandos DVC 

```cmd
dvc stage add -n prepro -d data/data.csv -d params.yaml -o data/clean_data.csv python src/preprocess.py data/data.csv data/clean_data.csv params.yaml

dvc stage add -n transform -d data/clean_data.csv -d params.yaml -o data/transformed_data.csv python src/transform.py data/clean_data.csv data/transformed_data.csv params.yaml

dvc stage add -n featureEngineer -d data/transformed_data.csv -d params.yaml -o data/top_features.csv python src/feature_engineer.py data/transformed_data.csv data/top_features.csv params.yaml

dvc stage add -n hiperparameters -d data/top_features.csv -d params.yaml python src/hiperparameters.py data/top_features.csv .\params.yaml

dvc stage add -n train -d data/top_features.csv -d params.yaml -o data/models.csv python src/train.py data/top_features.csv data/models.csv params.yaml

dvc stage add -n evaluate -d data/top_features.csv -d params.yaml -o data/results.csv python src/evaluate.py data/top_features.csv data/results.csv evaluation_metrics.json params.yaml
```

* Si existe un incoveniente en ejecutar la pipeline por favor verificar el archivo readme.md donde se menciona consideraciones generales para procesar la pipeline.

### Github Release
https://github.com/gdvr/gdvr-productDevelopment-lab1_24002907/releases/tag/v1.0.2

### Ejecucion
* git init
* dvc pull -f (si en dado caso falla el comando, colocar el archivo BK\data.csv manualmente en la carpeta data)
* dvc repro -f


### Metricas de evaluacion
|model|MAE|MSE|RMSE|CV MAE|R2 Score
|------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
LinearRegression_optimazed.pkl|836620.92|1208515174057.0|1099324.87|804538.35|0.6
RandomForest_optimazed.pkl|888593.53|1410382279372.28|1187595.17|854558.97|0.53
RandomForest_optuna.pkl|918346.2|1526738806246.56|1235612.73|869413.38|0.5
GradientBoosting_optimazed.pkl|883127.09|1484478180694.63|1218391.64|885426.67|0.51
bestModel_GradientBoosting.pkl|883256.91|1485043640308.72|1218623.67|886651.17|0.51
GradientBoosting_optuna.pkl|1022771.74|1895661143806.76|1376830.11|960286.28|0.37


El modelo ganador fue La regresion lineal que esta explica el 60% de la varianza de los datos y adicional tiene el mejor valor CV MAE que es de 804538.35.

### Optimizacion de parametros
Se realizo estudios utilizando optuna asi como RandomSearch and GridSearch pero aun asi los valores de optuna versus los valores de la busqueda mediente Random o Grid search tuviero un mejor desempeño.


