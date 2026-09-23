import pandas as pd

def features(df:pd.DataFrame)->pd.DataFrame:

    df = df.copy()

    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    df["Electrical"] = df["Electrical"].fillna(
        df["Electrical"].mode()[0]
    )

    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["OverallQual_GrLivArea"] = (
        df["OverallQual"] * df["GrLivArea"]
    )

    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    df["OverallQual_TotalSF"] = (
        df["OverallQual"] * df["TotalSF"]
    )

    df["MSSubClass"] = df["MSSubClass"].astype(str)

    return df