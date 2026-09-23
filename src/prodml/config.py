from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: str = "./data/train.csv"
    model_path: str = "./models"
    reports_path: str = "./reports"
    n_estimators: int = 220
    learning_rate: float = 0.2
    max_depth: int = 2
    random_state: int = 42
    test_size: float = 0.2


settings = Settings()
