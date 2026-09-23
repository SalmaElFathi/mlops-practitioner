import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

from prodml.data import load_data,split_data
from prodml.features import features
from prodml.config import DATA_PATH,REPORTS_PATH,MODEL_PATH,N_ESTIMATORS,MAX_DEPTH,LEARNING_RATE,RANDOM_STATE

df = load_data(DATA_PATH)
df_train, df_validation = split_data(df)

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

mask = (
    (X_train["GrLivArea"] >= lower)
    & (X_train["GrLivArea"] <= upper)
)

X_train = X_train[mask]
y_train = y_train[mask]


numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", encoder, categorical_features)
    ]
)

xgb = XGBRegressor(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE
)

model_final = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb)
    ]
)

model_final.fit(
    X_train,
    y_train
)

y_pred = model_final.predict(
    X_validation
)


rmse_final = np.sqrt(
    mean_squared_error(
        y_validation,
        y_pred
    )
)

mae_final = mean_absolute_error(
    y_validation,
    y_pred
)

print(f"RMSE : {rmse_final:.2f}")
print(f"MAE  : {mae_final:.2f}")

os.makedirs(
    REPORTS_PATH,
    exist_ok=True
)

with open(
    f"{REPORTS_PATH}/module-1.md",
    "w"
) as f:

    f.write(
        "## Module 1 - XGBoost\n\n"
    )

    f.write(
        f"- **RMSE (validation)** : "
        f"{rmse_final:.2f}\n"
    )

    f.write(
        f"- **MAE (validation)** : "
        f"{mae_final:.2f}\n"
    )


os.makedirs(
    MODEL_PATH,
    exist_ok=True
)

with open(
    f"{MODEL_PATH}/baseline.pkl",
    "wb"
) as f:

    pickle.dump(
        model_final,
        f
    )
