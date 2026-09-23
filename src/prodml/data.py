import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_validation = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    return df_train, df_validation
