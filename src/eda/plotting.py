import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv("../../data/train.csv")

train_df["Pclass"] = train_df["Pclass"].astype("int")

print(train_df.head(10))
print(train_df.dtypes)

is_plotted = {
    "Pclass": True
}

if __name__ == "__main__":
    if is_plotted["Pclass"]:
        pclass_surv = sns.histplot(train_df, x="Pclass", hue="Survived", multiple="dodge")
        pclass_sex = sns.histplot(train_df, x="Pclass", hue="Sex", multiple="dodge")

        g = sns.FacetGrid(train_df, row="Sex")
        g.map(sns.histplot, "Pclass")

    plt.show()
