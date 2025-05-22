
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv("data/processed/sales_clean.csv")
X = df[['price', 'promo']]
y = df['sales_volume']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['price']),
    ('cat', OneHotEncoder(), ['promo'])
])
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow tracking
mlflow.set_experiment("RetailDemandForecast")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    # Predict + evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mae)

    # Log artifacts
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(pipeline, "model")

    # Save locally
    joblib.dump(pipeline, "model.joblib")
