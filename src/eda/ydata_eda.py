import pandas as pd
from ydata_profiling import ProfileReport

if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")

    type_schema = {
        "Survived": "categorical",
        "Sex": "categorical",
        "Embarked": "categorical"
    }

    profile = ProfileReport(train_df, title="Basic EDA", type_schema=type_schema)
    profile.to_file("your_report.html")
