# MLOps Practitioner — House Price Prediction

Python package that predicts a house's sale price (Ames Housing dataset) from its features: living area, neighborhood, overall quality, etc. XGBoost model wrapped in a scikit-learn pipeline.

## Commands

pip install -e ".[dev]"                            # install
ruff check src tests && black --check src tests    # lint
python -m prodml.train                             # train

## Structure

- `src/prodml/data.py` — load CSV, train/validation split
- `src/prodml/features.py` — feature engineering (missing values, derived columns)
- `src/prodml/train.py` — train the pipeline and persist the model
- `src/prodml/predict.py` — load model and predict (`SalePricePredictor` class)
- `src/prodml/config.py` — configuration (paths, hyperparameters) via pydantic-settings
- `src/prodml/utils.py` — `@timed` decorator

## Results (baseline)

RMSE: 29585.61
MAE:  16339.96
