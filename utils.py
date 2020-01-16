""""
You can see many helper functions in this files.
For instance:
1. Params class to load and store hyper parameters.
2. Get your data set, concatenate and divide your data set. (More than 1 file need them.)
3. Get missing values and save the data features to markdown. (The reason why they are written in this file is same.)
"""

import json
import os

import pandas as pd
from tabulate import tabulate


class Params:
    def __init__(self, json_path):
        # 根据json_path跟新对应的json参数
        with open(json_path, "r") as f:
            self.__dict__.update(json.load(f))

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, json_path):
        with open(json_path, "r") as f:
            self.__dict__.update(json.load(f))

    @property
    def dict(self):
        # 在整个类的声明里面我们使用self.__dict__
        # 但是我们对外确实使用了dict的只读属性
        return self.__dict__


def GetDataSet(filename: str):
    """ Get test set, train set x and y """
    # Read csv files
    # Assign our data set name attribute to simplify our code
    df_train_set = pd.read_csv(filename + "/train.csv")
    df_train_set.name = "train"
    df_test_set = pd.read_csv(filename + "/test.csv")
    df_test_set.name = "test"
    # Combine train set and test set
    df_all = ConcatDF(df_train_set, df_test_set)
    df_all.name = "all"
    # Remember to return the train set and test set for the next step
    return df_train_set, df_test_set, df_all


def ConcatDF(train_set, test_set):
    """
    Concatenate train set and test set,
    So our filling data won't over-fit on the train set.
    """
    return pd.concat([train_set, test_set], sort=True).reset_index(drop=True)


def DivideDF(all_data):
    """ Divide the data set that concatenated from train set and test set """
    return all_data.iloc[:890], all_data.iloc[891:].drop("Survived", axis=1)


def GetMissingValues(data_set: pd.DataFrame):
    """ Show missing numbers if it exists """
    # Get missing features and missing line counts
    missing_features = []
    missing_line_counts = []
    for column in data_set.columns:
        missing_line_count = data_set[column].isnull().sum()
        if missing_line_count != 0:
            missing_features.append(column)
            missing_line_counts.append(missing_line_count)
    missing_rate = [item / data_set.shape[0] for item in missing_line_counts]
    # Create given data set
    result = pd.DataFrame({"features": missing_features,
                           "missing lines": missing_line_counts,
                           "missing rate": missing_rate},
                          dtype="int64")
    return result


def Save2Markdown(data_set, dir_path):
    # Get all information
    numerical_feature = data_set.describe()  # Numerical data
    categorical_feature = data_set.describe(include=["O"])  # categorical data
    missing_conditions = GetMissingValues(data_set)
    information_dict = {
        "Numerical feature": numerical_feature,
        "Categorical feature": categorical_feature,
        "Missing conditions": missing_conditions
    }
    # Save to relevant markdown files
    save_path = os.path.join(dir_path, data_set.name + "_set_analysis.md")
    with open(save_path, "w+") as f:  # Won't continue to append when rerunning
        for info_key, info_value in information_dict.items():
            f.write("# " + info_key + "\n")
            table = tabulate(info_value, headers="keys", tablefmt="pipe")
            f.write(table)
            f.write("\n\n")
