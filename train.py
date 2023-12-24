import json

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

PATH_DATA = "data/all_v2.csv"
PATH_UNIQUE_VALUES = 'data/unique_values.json'
PATH_MODEL = 'models/lr_pipeline.sav'
drop_cols = ['date', 'time', 'geo_lat', 'geo_lon', 'region']
categorical_features = ['building_type', 'object_type']
numeric_features = ['level', 'levels', 'rooms', 'area', 'kitchen_area']
passthrough_feats = ['price']

df = pd.read_csv(PATH_DATA)
df = df.drop(columns=drop_cols)

df = df[df.price.between(df.price.quantile(0.05), df.price.quantile(0.95))]
df = df[df.area.between(df.area.quantile(0.05), df.area.quantile(0.95))]
df = df[df.rooms > -2]

y = df['price']
X = df.drop(columns="price", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
)

clf = make_pipeline(preprocessor, LinearRegression())

clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)

print(mean_absolute_error(y_test, y_prediction))

joblib.dump(clf, PATH_MODEL)

dict_unique = {key: X[key].unique().tolist() for key in X.columns}

with open(PATH_UNIQUE_VALUES, 'w') as file:
    json.dump(dict_unique, file)