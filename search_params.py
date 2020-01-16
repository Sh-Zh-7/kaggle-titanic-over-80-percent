"""
Search all model's best hyper parameters, independent from main.py as well.
The model we used here is only random forest classifier.
So we just need to search random forest's best hyper parameters and save them for main to use.
"""

from sklearn.model_selection import GridSearchCV

from main import *
from utils import *

if __name__ == "__main__":
    # Get all data sets
    _, __, df_all = GetDataSet("./data")

    # Filling missing values
    FillingMissingValues(df_all)

    # Feature engineering
    df_train_set, df_test_set = FeatureEngineer(df_all)
    features = df_train_set.columns

    # Machine learning——pre-processing
    X_train = StandardScaler().fit_transform(df_train_set.drop("Survived", axis=1))
    y_train = df_train_set['Survived']
    X_test = StandardScaler().fit_transform(df_test_set)

    # Search best params
    tuned_parameters = Params("./models/rf_params_list.json").dict
    rfc = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5)
    rfc.fit(X_train, y_train)

    # Save best params
    with open("./models/rf_best_params.json", "w") as f:
        json.dump(rfc.best_params_, f, indent=4)
