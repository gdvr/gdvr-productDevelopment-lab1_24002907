
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

def createPipeline(categoricals, models):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categoricals)
        ],
        remainder='passthrough' 
    )

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