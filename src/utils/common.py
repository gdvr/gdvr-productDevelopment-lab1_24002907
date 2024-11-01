import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def splitValuesForModel(X,y, TEST_SIZE, VALIDATE_SIZE,RANDOM_STATE):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,  y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VALIDATE_SIZE, random_state=RANDOM_STATE)

    print(f"Training set class distribution:\n{X_train.shape}-{y_train.shape}")
    print(f"Validation set class distribution:\n{X_val.shape}-{y_val.shape}")
    print(f"Test set class distribution:\n{X_test.shape}-{y_test.shape}")

    return X_train, y_train,X_test,y_test,X_val, y_val

def categorizeColumns(dataset):
    continuas, discretas, categoricas = __get_variables_scale_type(dataset)
    print(f"# Continuas: {len(continuas)}, values: {', '.join(continuas)}")
    print(f"# Discretas: {len(discretas)}, values: {', '.join(discretas)}")
    print(f"# Categoricas: {len(categoricas)}, values: {', '.join(categoricas)}")

    return continuas, discretas, categoricas

def createPipeline(categoricals, numerics, models):
    preprocessor = createPreprocesor(categoricals,numerics)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', models)
    ])

    return model_pipeline

#Funcion que detecta los valores nulos
def detectInvalidValues(dataset):    
    columnas = dataset.columns
    for col in columnas:        
        porcentaje = dataset[col].isnull().mean()
        if porcentaje > 0:               
            print(f"Percentage of null values for {col}: {porcentaje}%")            
        else:
            print(f"No invalid data for {col}")           

def handlingEmptyValues(dataset,cols):
    print(f"Fill the empty values with mean for cols: {', '.join(cols)}")
    dataset[cols] = dataset[cols].apply(lambda col: col.fillna(col.mean()), axis=0)
    return dataset

#Funcion que permite clasificar las columnas en categoricas, discretas y continuas
def __get_variables_scale_type(dataset):
    columnas = dataset.columns
    categoricas = []
    continuas = []
    discretas = []

    for col in columnas:
        print(col)
        col_type=dataset[col].dtype
        
        if(col_type == 'object' or col_type == 'category'):
            categoricas.append(col)
        elif((col_type =='int64' or col_type =='int32') or (col_type =='float64' or col_type =='float32')):
            n = len(dataset[col].unique())
            if(n > 30):
                continuas.append(col)
            else:
                discretas.append(col)
    
    return continuas, discretas, categoricas     

def rf_objective(trial, X_train, y_train, X_val, y_val, random_state):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse


def gb_objective(trial, X_train, y_train, X_val, y_val, random_state):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    
    gb = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        validation_fraction=0.2,  # 20% of training data for validation
        n_iter_no_change=10,      # Stop if no improvement for 10 rounds
        tol=1e-4                  # Tolerance level for early stopping
    )
    
    gb.fit(X_train, y_train)
    preds = gb.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse



def hyperparameter_search(model, param_grid, X_train,y_train, cv, search_type):
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    
    search.fit(X_train, y_train)
    return search.best_estimator_, -search.best_score_

def chooseBestHiperparameters(X_train,y_train, cv, random_state):
    models_and_params = {
        'RandomForest': (RandomForestRegressor(random_state=random_state), {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }),
        'GradientBoosting': (GradientBoostingRegressor(random_state=random_state), {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10]
        }),
        'LinearRegression': (LinearRegression(), {
            'fit_intercept': [True, False]
        })
    }

    best_models = {}
    for mode in ['grid','random']:
        for model_name, (model, param_grid) in models_and_params.items():
            if(model_name == 'LinearRegression' and mode == 'random'):
                print('Skip this hiperparameters search')
                continue
            print(f"Running search for {model_name} and {mode} search...")
            best_model, best_score = hyperparameter_search(model, param_grid, X_train,y_train,cv,mode)
            best_models[model_name] = (best_model, best_score)
            print(f"{model_name} best score: {best_score:.4f}")

    # Compare models and select the best one
    best_model_name = min(best_models, key=lambda k: best_models[k][1])
    best_model, best_score = best_models[best_model_name]

    joblib.dump(best_model, f"models/bestModel_{best_model_name}.joblib")

    print(f"\nBest model: {best_model_name} with score: {best_score:.4f}")
    print(f"Best model details:\n{best_model}")

    return best_model

def createPreprocesor(categoricals, numerics):
    one_hot_encoder = OneHotEncoder()
    preprocessor = ColumnTransformer(
        transformers=[           
            ('num', 'passthrough', numerics),
            ('cat', one_hot_encoder, categoricals),
        ]
    )

    return preprocessor