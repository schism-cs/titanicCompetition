from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd


def feature_synthesis(df):
    # Adding a FamilySize feature
    df["FamilySize"] = df["SibSp"] + df["Parch"]

    # Adding an isAlone feature, in case the passenger doesn't have family on board
    df = df.assign(IsAlone='false')
    df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0), 'IsAlone'] = 'true'

    # Binning Age and Fare into 2 new synthetic features
    X = df["Age"].array.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform', subsample=None)
    est.fit(X)
    Xt = est.transform(X)
    df["BinnedAge"] = Xt
    df["BinnedAge"] = df["BinnedAge"].astype("str").astype("category")

    X = df["Fare"].array.reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', subsample=None)
    est.fit(X)
    Xt = est.transform(X)
    df["BinnedFare"] = Xt
    df["BinnedFare"] = df["BinnedFare"].astype("str").astype("category")

    return df
