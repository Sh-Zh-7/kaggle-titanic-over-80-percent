"""
Do all the machine learning procedures require to do.
The procedures are as following shows:
1. Get train set and test set, concatenate them to better fill the missing values
2. EDA, see how many the missing values are and fill them, watch distribution map, corr map and so on.
3. FE, not only add new features and delete old features, but also convert them to digits.
4. ML, use the best model that we have find to fit and predict the target.
(You can draw importance plot or others to do further work)
5. Use the pandas api to output the result.
"""

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from eda import FillingMissingValues
from plot import *
from train import Train
from utils import *

warnings.filterwarnings("ignore")
sns.set()


def FeatureEngineer(df_all: pd.DataFrame):
    """ Dirt hand feature engineering """
    df_all.drop("PassengerId", axis=1, inplace=True)
    df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
    df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    idx = df_all[df_all['Deck'] == 'T'].index
    df_all.loc[idx, 'Deck'] = 'A'
    df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
    df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
    df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
    df_all['Age'] = pd.qcut(df_all['Age'], 10)
    df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
                  11: 'Large'}
    df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)
    df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')
    df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df_all['Is_Married'] = 0
    df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1
    df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                                              'Miss/Mrs/Ms')
    df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
                                              'Dr/Military/Noble/Clergy')
    non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']
    df_train_set, df_test_set = DivideDF(df_all)
    dfs = [df_train_set, df_test_set]
    cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
    for df in dfs:
        for feature in non_numeric_features:
            df[feature] = LabelEncoder().fit_transform(df[feature])

    drop_cols = ['Family_Size', 'Cabin', "Parch", "SibSp", "Name", 'Ticket']
    df_train_set.drop(columns=drop_cols, inplace=True)
    df_test_set.drop(columns=drop_cols, inplace=True)
    df_train_set = pd.get_dummies(df_train_set, columns=cat_features)
    df_test_set = pd.get_dummies(df_test_set, columns=cat_features)
    return df_train_set, df_test_set


def GetModel(path):
    """ Return best model that have searched """
    params = Params(os.path.join(path, "rf_best_params.json"))
    model = RandomForestClassifier(**params.dict)
    return model


if __name__ == "__main__":
    # Get all data sets
    _, __, df_all = GetDataSet("./data")

    # Filling missing values
    FillingMissingValues(df_all)

    # Feature engineering
    df_train_set, df_test_set = FeatureEngineer(df_all)
    features = df_test_set.columns

    # Machine learning
    # Pre-processing
    X_train = StandardScaler().fit_transform(df_train_set.drop("Survived", axis=1))
    y_train = df_train_set['Survived']
    X_test = StandardScaler().fit_transform(df_test_set)
    # Training
    model = GetModel("./models")  # Load the best model that find in search_params.py
    fprs, tprs, scores, importance = Train(model, X_train, y_train)

    # Draw roc curve and save
    PlotROCCurve(fprs, tprs)
    plt.show()
    plt.savefig("./analysis/roc_curve.png")

    # Draw feature importance map and save
    importance['Mean_Importance'] = importance.mean(axis=1)
    PlotImportance(features, importance["Mean_Importance"])
    plt.show()
    plt.savefig("./analysis/features_importance.png")

    # Outputs
    outputs = model.predict(X_test)
    result = pd.DataFrame({"PassengerId": range(892, 1310), "Survived": outputs.astype(int)})
    result.to_csv("./result.csv", index=False)
