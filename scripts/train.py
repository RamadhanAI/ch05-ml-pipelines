
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

df = pd.read_csv("data/processed/sales_clean.csv")
X = df[['price', 'promo']]
y = df['sales_volume']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['price']),
    ('cat', OneHotEncoder(), ['promo'])
])
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model.joblib")
