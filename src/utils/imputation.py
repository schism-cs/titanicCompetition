import pandas as pd
from sklearn.impute import KNNImputer


def age_imputation(df, impute_fare=False):
    features = df.columns.values.tolist()

    features.remove("PassengerId")
    if "Survived" in features:
        features.remove("Survived")
    features.remove("Name")
    features.remove("Ticket")
    features.remove("Cabin")
    features.remove("Embarked")
    features.remove("Ticket_number")
    features.remove("Ticket_item")

    cleaned_df = df[features]
    cleaned_df.loc[:, 'Sex'] = cleaned_df["Sex"].replace({'male': 0, 'female': 1})

    imputer = KNNImputer(n_neighbors=2, copy=True)
    imputed = imputer.fit_transform(cleaned_df)

    imp_df = pd.DataFrame(imputed)

    df["Age"] = imp_df[2]

    if impute_fare:
        df["Fare"] = imp_df[5]

    return df, imp_df
