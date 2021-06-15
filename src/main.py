"""This script is used for training and testing a given set of Machine Learning algorithms on a given input csv
Usage:
    my_script.py <data> <output_folder> [options]
Arguments:
    <data>                     Path of an input csv file
    <output_folder>            A folder to collect all the results from training and testing algorithms
"""

#TO DO:
#1. Use parameters instead of global var
#2. Show algo's advancement
#3. Save gridsearch results in a text file
#4. Print: size of subsample, number of new categories..

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
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

DATA = Path('../data/output_api_entreprise.csv')
OUTPUT_FOLDER = Path('../output/fc')


def main():
    # 1. Read data and drop useless columns
    data = pd.read_csv(DATA)
    data = data.drop(columns=['id', 'siret', 'categorie_juridique', 'activite_principale', 'instruction_comment',
                              'fondement_juridique_url'])
    # 2. Missing values imputation: create a new category for missing values
    data = impute_nans(data)
    # 3. Aggregate categorical variables because of high cardinality
    cat_variables = ['target_api', 'categorie_juridique_label', 'activite_principale_label']
    data = aggregate_cat(data, cat_variables)
    print()
    # 4. Encoders (categorical variables & text)
    one_hot_enc = OneHotEncoder(handle_unknown='ignore')
    label_enc = preprocessing.LabelEncoder()
    text_enc = TfidfVectorizer()
    columns_trans = make_column_transformer((one_hot_enc, cat_variables), (text_enc, 'nom_raison_sociale'),
                                            (text_enc, 'intitule'), (text_enc, 'fondement_juridique_title'),
                                            (text_enc,'description'))
    # 5. Train/test splitting
    y = data['status'].values
    y = label_enc.fit_transform(y)
    X = data.drop(columns=['status'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # 6. Train and test algorithms
    algorithms = [LogisticRegression(),RandomForestClassifier(),XGBClassifier(), CatBoostClassifier(),
                  SVC(), MLPClassifier(), lgb.LGBMClassifier(), DecisionTreeClassifier()]
    algorithms_names = ['LogisticRegression','RandomForestClassifier','XGBClassifier', 'CatBoostClassifier',
                  'SVC', 'MLPClassifier', 'lgb.LGBMClassifier', 'DecisionTreeClassifier']
    for algorithm,algo_name in zip(algorithms,algorithms_names):
        algo = algorithm
        pipe = make_pipeline(columns_trans,algo)
        #algo_name = str(algorithm).partition('(')[0]
        param_grid = algorithms_grid[algo_name]
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
        grid.fit(X_train, y_train)
        prediction = grid.predict(X_test)
        report = classification_report(y_test, prediction,output_dict=True)
        pd.DataFrame(report).to_csv(f'{OUTPUT_FOLDER}/classif_report_{algo_name}.csv')
        confusion = confusion_matrix(y_test, prediction)
        sns.heatmap(confusion, annot=True, vmin=0, vmax=len(y_test),cmap='Blues', fmt='g')
        plt.savefig(f'{OUTPUT_FOLDER}/confusion_matrix_{algo_name}.png')



def impute_nans(data):
    missing_values_imp = SimpleImputer(strategy='constant', fill_value='missing')
    missing_cols = ['categorie_juridique_label', 'activite_principale_label', 'nom_raison_sociale', 'description',
                    'intitule', 'fondement_juridique_title']
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
                                            "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                                            "columntransformer__tfidfvectorizer-1__max_features": [None, 200]},
                   'RandomForestClassifier': {"randomforestclassifier__n_estimators": [100, 200, 400],
                                                "randomforestclassifier__max_depth": [4, 6, 8, 10, 12],
                                                "randomforestclassifier__max_features": ["auto", 0.2],
                                                "randomforestclassifier__min_samples_split": [10],
                                                "randomforestclassifier__random_state": [64],
                                                "randomforestclassifier__class_weight": ['balanced', {0: .3, 1: .7},
                                                                                         {0: .4, 1: .6}],
                                                "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                                                "columntransformer__tfidfvectorizer-1__max_features": [None, 200]
                                                },

                   'XGBClassifier': {
                       "xgbclassifier__objective": ["binary:logistic"],
                       "xgbclassifier__eval_metric": ["logloss"],
                       "xgbclassifier__eta": [0.05, 0.075, 0.1, 0.3],
                       "xgbclassifier__max_depth": [4, 5, 6],
                       "xgbclassifier__min_child_weight": [1, 2],
                       "xgbclassifier__subsample": [0.5, 1.0],
                       "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                       "columntransformer__tfidfvectorizer-1__max_features": [None, 200]
                   },
                   'CatBoostClassifier': {
                       "catboostclassifier__learning_rate": [0.1],
                       "catboostclassifier__depth": [6],
                       "catboostclassifier__rsm": [0.9],  # random subspace method
                       "catboostclassifier__subsample": [1],  # random subspace method
                       "catboostclassifier__min_data_in_leaf": [15],
                       "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                       "columntransformer__tfidfvectorizer-1__max_features": [None, 200]},
                   'SVC': {"svc__C": [0.1, 1.0],
                             "svc__gamma": ['auto'],
                             "svc__kernel": ['rbf', 'linear'],
                             "svc__class_weight": ['balanced', None, {0: .4, 1: .6}, {0: .3, 1: .7}, {0: .7, 1: .3}],
                             "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                             "columntransformer__tfidfvectorizer-1__max_features": [None, 200]},
                   'MLPClassifier': {"mlpclassifier__activation": ["relu", "tanh"],
                                       "mlpclassifier__solver": ["adam"],
                                       "mlpclassifier__alpha": [0.0001, 0.001],
                                       "mlpclassifier__batch_size": [100],
                                       "mlpclassifier__learning_rate_init": [0.0001, 0.001],
                                       "mlpclassifier__random_state": [64],
                                       "mlpclassifier__early_stopping": [True],
                                       "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                                       "columntransformer__tfidfvectorizer-1__max_features": [None, 200]
                                       },
                   'lgb.LGBMClassifier': {
                       "lgbmclassifier__objective": ["binary"],
                       "lgbmclassifier__metric": ["binary_logloss"],
                       "lgbmclassifier__num_leaves": [3, 7, 15, 31],
                       "lgbmclassifier__learning_rate": [0.05, 0.075, 0.1, 0.15],
                       "lgbmclassifier__feature_fraction": [0.8, 0.9, 1.0],
                       "lgbmclassifier__bagging_fraction": [0.8, 0.9, 1.0],
                       "lgbmclassifier__min_data_in_leaf": [5, 10, 15],
                       "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                       "columntransformer__tfidfvectorizer-1__max_features": [None, 200]
                   },
                   'DecisionTreeClassifier': {
                       "decisiontreeclassifier__criterion": ["gini", "entropy"],
                       "decisiontreeclassifier__max_depth": [2, 3, 4],
                       "columntransformer__tfidfvectorizer-1__min_df": [1, 2],
                       "columntransformer__tfidfvectorizer-1__max_features": [None, 200]
                   }}

if __name__ == "__main__":
    main()
