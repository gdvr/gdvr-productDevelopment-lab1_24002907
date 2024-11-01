# src/preprocess.py
import pandas as pd
import sys
import optuna
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from utils.common import chooseBestHiperparameters, gb_objective, rf_objective, splitValuesForModel

def preprocess(features, target):    
    df_training =  pd.read_csv("data/X_train.csv")
    df_test =  pd.read_csv("data/X_test.csv")
    df_val =  pd.read_csv("data/X_val.csv")

    X_train = df_training[features]
    y_train = df_training[target]
    X_test = df_test[features]
    y_test = df_test[target]
    X_val = df_val[features]
    y_val = df_val[target]

    random_state = params['train']['RANDOM_STATE']

    one_hot_encoder  = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), params['categoricas'])
        ],
        remainder='passthrough' 
    )

    X_train_encoded = one_hot_encoder.fit_transform(X_train)
    X_val_encoded = one_hot_encoder.transform(X_val)
    X_test_encoded = one_hot_encoder.transform(X_test)

    #model = chooseBestHiperparameters(X_train_encoded,y_train,params['train']['CV'],params['train']['RANDOM_STATE'])

    data =  pd.read_csv("data/clean_data.csv")
    X = data[features]
    y = data[target]

   
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', one_hot_encoder, params['categoricas']),
            ('num', 'passthrough', params['continuas']+params['discretas'])
        ]
    )

    # Create a pipeline that first transforms the data and then fits the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=params["train"]["RANDOM_STATE"]))
    ])

    # Fit the pipeline to the data
    pipeline.fit(X, y)

    # Get feature importances
    importances = pipeline.named_steps['model'].feature_importances_

    # Get feature names after One-Hot Encoding
    ohe_feature_names = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(params['categoricas'])
    all_feature_names =  params['continuas']+params['discretas'] + list(ohe_feature_names)

    # Create a DataFrame with feature importances
    importance_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    # Save the top important features
    top_n_features = params["feature_engineering"]["top_n_features"]
    top_features = importance_df.head(top_n_features)
    top_features.to_csv("data/top_features.csv", index=False)
    
    # Transform the original dataset with One-Hot Encoding
    X_transformed = preprocessor.fit_transform(X)

    # Create a DataFrame with the transformed dataset
    transformed_feature_names =  params['continuas']+params['discretas'] + list(ohe_feature_names)
    X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_feature_names)

    # Combine the transformed features with the target variable
    final_dataset = pd.concat([X_transformed_df, y.reset_index(drop=True)], axis=1)

    # Save the final dataset with One-Hot Encoded columns
    final_dataset.to_csv("data/transformed_data.csv", index=False)
    
    most_important_features = final_dataset.drop(columns=[target], errors='ignore').columns.to_list()

    X = final_dataset.drop(columns=["price"], errors='ignore')[most_important_features]
    y = final_dataset[target]
    

    X_train, y_train,X_test,y_test,X_val, y_val = splitValuesForModel(X,y,params['train']['TEST_SIZE'],params['train']['VALIDATE_SIZE'],params['train']['RANDOM_STATE'])
  
    
    rf_study = optuna.create_study(direction="minimize")
    rf_study.optimize(lambda trial: rf_objective(trial, X_train_encoded, y_train, X_val_encoded, y_val, random_state), n_trials=50)
    print("Best Random Forest parameters:", rf_study.best_params)
    params["RandomForest"] = rf_study.best_params

    gb_study = optuna.create_study(direction="minimize")
    gb_study.optimize(lambda trial: gb_objective(trial, X_train_encoded, y_train, X_val_encoded, y_val, random_state), n_trials=50)
    print("Best Gradient Boosting parameters:", gb_study.best_params)
    params["GradientBoosting"] = gb_study.best_params

    with open("params.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    params_file = sys.argv[1]

    with open(params_file) as f:
        params = yaml.safe_load(f)
    
    print(params)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    

    preprocess(features, target)
