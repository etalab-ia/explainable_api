# Script to only train and test on categorical variables

import glob
import os
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

DATA = Path('../data')
OUTPUT_FOLDER = Path('../output/output_no_text')


def main():
    # 1. Read data, drop useless columns and create a proper output folder. Also drop text columns
    list_csvs = glob.glob(os.path.join(DATA, "*.csv"))
    for dataset in list_csvs:
        name_output = re.search("/data/(.*?)\.csv", dataset).group(1)
        if not OUTPUT_FOLDER.joinpath(name_output).exists():
            os.mkdir(OUTPUT_FOLDER.joinpath(name_output))
        data = pd.read_csv(dataset)
        data = data.drop(columns=['id', 'siret', 'categorie_juridique', 'activite_principale', 'instruction_comment',
                                  'fondement_juridique_url','nom_raison_sociale','intitule','fondement_juridique_title','description'])
        print(f"Now treating dataset {name_output}")
        print(f"This dataset has {len(data)} rows")
        # 2. Missing values imputation: create a new category for missing values
        data = impute_nans(data)
        # 3. Aggregate categorical variables because of high cardinality
        cat_variables = ['target_api', 'categorie_juridique_label', 'activite_principale_label']
        data = aggregate_cat(data, cat_variables)
        print(
            f"The new number of categories for categorie_juridique_label is {data['categorie_juridique_label'].nunique()}")
        print(
            f"The new number of categories for activite_principale_label is {data['activite_principale_label'].nunique()}")
        # 4. Encoders (categorical variables & text)
        one_hot_enc = OneHotEncoder(handle_unknown='ignore')
        label_enc = preprocessing.LabelEncoder()
        columns_trans = make_column_transformer((one_hot_enc, cat_variables))
        # 5. Train/test splitting
        y = data['status'].values
        y = label_enc.fit_transform(y)
        X = data.drop(columns=['status'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)
        # 6. Train and test algorithms
        algorithms = [LogisticRegression(), RandomForestClassifier(), XGBClassifier(), CatBoostClassifier(),
                      SVC(), DecisionTreeClassifier()]
        algorithms_names = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier', 'CatBoostClassifier',
                            'SVC', 'DecisionTreeClassifier']
        for algorithm, algo_name in zip(algorithms, algorithms_names):
            algo = algorithm
            pipe = make_pipeline(columns_trans, algo)
            param_grid = algorithms_grid[algo_name]
            grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
            print(f"Now starting to fit : {algo_name}")
            grid.fit(X_train, y_train)
            print("Best estimator:\n{}".format(grid.best_estimator_))
            prediction = grid.predict(X_test)
            report = classification_report(y_test, prediction, output_dict=True)
            pd.DataFrame(report).to_csv(f'{OUTPUT_FOLDER}/{name_output}/classif_report_{algo_name}.csv')
            confusion = confusion_matrix(y_test, prediction)
            sns.heatmap(confusion, annot=True, vmin=0, vmax=len(y_test), cmap='Blues', fmt='g')
            plt.savefig(f'{OUTPUT_FOLDER}/{name_output}/confusion_matrix_{algo_name}.png')
            plt.close()
            print(f"Output saved in {OUTPUT_FOLDER}/{name_output}")


def impute_nans(data):
    missing_values_imp = SimpleImputer(strategy='constant', fill_value='missing')
    missing_cols = ['categorie_juridique_label', 'activite_principale_label']
    data[missing_cols] = missing_values_imp.fit_transform(data[missing_cols])
    return data


def aggregate_cat(data, cat_variables):
    for variable in cat_variables:
        if data[variable].nunique() > 20:
            for category in data[variable].unique():
                if data[variable].value_counts()[category] / len(data) <= 0.01:
                    data.loc[data[variable] == category, variable] = 'Autre'
    return data


algorithms_grid = {'LogisticRegression': {"logisticregression__C": np.arange(0.4, 1.5, 0.2),
                                          "logisticregression__class_weight": ['balanced', {0: .3, 1: .7},
                                                                               {0: .4, 1: .6}, 'auto'],
                                          "logisticregression__max_iter": [300]},
                   'RandomForestClassifier': {"randomforestclassifier__n_estimators": [100, 200, 400],
                                              "randomforestclassifier__max_depth": [4, 6, 8, 10, 12, 15],
                                              "randomforestclassifier__max_features": ["auto"],
                                              "randomforestclassifier__min_samples_split": [10],
                                              "randomforestclassifier__random_state": [64],
                                              "randomforestclassifier__class_weight": ['balanced', {0: .3, 1: .7},
                                                                                       {0: .4, 1: .6}],
                                              },

                   'XGBClassifier': {
                       "xgbclassifier__objective": ["binary:logistic"],
                       "xgbclassifier__eval_metric": ["logloss"],
                       "xgbclassifier__eta": [0.05, 0.075, 0.1, 0.3],
                       "xgbclassifier__max_depth": [4, 5, 6],
                       "xgbclassifier__min_child_weight": [1, 2],
                       "xgbclassifier__subsample": [0.5, 1.0],

                   },
                   'CatBoostClassifier': {
                       "catboostclassifier__learning_rate": [0.1],
                       "catboostclassifier__depth": [6],
                       "catboostclassifier__rsm": [0.9],  # random subspace method
                       "catboostclassifier__subsample": [1],  # random subspace method
                       "catboostclassifier__min_data_in_leaf": [15],

                   },
                   'SVC': {"svc__C": [0.1, 1.0],
                           "svc__gamma": ['auto'],
                           "svc__kernel": ['rbf', 'linear'],
                           "svc__class_weight": ['balanced', None, {0: .4, 1: .6}, {0: .3, 1: .7}, {0: .7, 1: .3}],


                           },
                   'lgb.LGBMClassifier': {
                       "lgbmclassifier__objective": ["binary"],
                       "lgbmclassifier__metric": ["binary_logloss"],
                       "lgbmclassifier__num_leaves": [3, 7, 15, 31],
                       "lgbmclassifier__learning_rate": [0.05, 0.075, 0.1, 0.15],
                       "lgbmclassifier__feature_fraction": [0.8, 0.9, 1.0],
                       "lgbmclassifier__bagging_fraction": [0.8, 0.9, 1.0],
                       "lgbmclassifier__min_data_in_leaf": [5, 10, 15],
                   },
                   'DecisionTreeClassifier': {
                       "decisiontreeclassifier__criterion": ["gini", "entropy"],
                       "decisiontreeclassifier__max_depth": np.arange(2, 30, 1),

                   }}

if __name__ == "__main__":
    main()