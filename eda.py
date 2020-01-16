"""
EDA, independent from main.py
Notice we have to get the data set and fill the missing values as well.
So you can see the similar part code that you have already seen in the main.py.
"""

from plot import *
from utils import *


def EDA(df_all):
    """ Overview all your features """
    df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "feature1", "level_1": " feature2", 0: "corr eff"}, inplace=True)
    df_all.drop("PassengerId", inplace=True, axis=1)


def FillingMissingValues(df_all):
    # Fill age's missing values
    df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    # Fill embarked's missing values
    df_all['Embarked'] = df_all['Embarked'].fillna('S')
    # Fill fare's missing values
    med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df_all['Fare'] = df_all['Fare'].fillna(med_fare)


def DrawHeatMap(data_set):
    """ Draw all feature's heat map(encapsulate plot function) """
    PlotHeatMapRectangle(data_set)
    plt.show()
    plt.savefig("./analysis/{}_set_heat_map.png".format(data_set.name))


if __name__ == "__main__":
    # Get all data sets
    df_train_set, df_test_set, df_all = GetDataSet("./data")
    # Get numerical and categorical feature's information and save to markdown
    Save2Markdown(df_train_set, "./analysis")
    Save2Markdown(df_test_set, "./analysis")
    # Filling Missing values
    FillingMissingValues(df_all)
    # Draw heat maps
    DrawHeatMap(df_train_set)
    DrawHeatMap(df_test_set)
