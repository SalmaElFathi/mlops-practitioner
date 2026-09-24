import logging
import os
import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import features
from prodml.logging_conf import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

df = load_data(settings.data_path)
df_train, df_validation = split_data(df, settings.test_size, settings.random_state)

df_train = features(df_train)
df_validation = features(df_validation)

y_train = df_train["SalePrice"]
X_train = df_train.drop(["SalePrice", "Id"], axis=1)

y_validation = df_validation["SalePrice"]
X_validation = df_validation.drop(["SalePrice", "Id"], axis=1)

Q1 = X_train["GrLivArea"].quantile(0.25)
Q3 = X_train["GrLivArea"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

mask = (X_train["GrLivArea"] >= lower) & (X_train["GrLivArea"] <= upper)

X_train = X_train[mask]
y_train = y_train[mask]


numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X_train.select_dtypes(include=["object"]).columns

encoder = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", encoder, categorical_features),
    ]
)

xgb = XGBRegressor(
    n_estimators=settings.n_estimators,
    learning_rate=settings.learning_rate,
    max_depth=settings.max_depth,
    random_state=settings.random_state,
)

model_final = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb)])

model_final.fit(X_train, y_train)

y_pred = model_final.predict(X_validation)


rmse_final = np.sqrt(mean_squared_error(y_validation, y_pred))

mae_final = mean_absolute_error(y_validation, y_pred)

logger.info(f"RMSE : {rmse_final:.2f}")
logger.info(f"MAE  : {mae_final:.2f}")

os.makedirs(settings.reports_path, exist_ok=True)

with open(f"{settings.reports_path}/module-1.md", "w") as f:

    f.write("## Module 1 - XGBoost\n\n")

    f.write(f"- **RMSE (validation)** : " f"{rmse_final:.2f}\n")

    f.write(f"- **MAE (validation)** : " f"{mae_final:.2f}\n")


os.makedirs(settings.model_path, exist_ok=True)

with open(f"{settings.model_path}/baseline.pkl", "wb") as f:

    pickle.dump(model_final, f)
