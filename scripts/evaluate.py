
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

model = joblib.load("model.joblib")
df = pd.read_csv("data/processed/sales_clean.csv")
X = df[['price', 'promo']]
y_true = df['sales_volume']
y_pred = model.predict(X)
print("MAE:", mean_absolute_error(y_true, y_pred))
