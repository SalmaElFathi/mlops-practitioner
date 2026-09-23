import pandas as pd
from sklearn.model_selection import train_test_split
from prodml.config import TEST_SIZE

def load_data(path:str)->pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_data(df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    df_train, df_validation = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=42
    )

    return df_train, df_validation
