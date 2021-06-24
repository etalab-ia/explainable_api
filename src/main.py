"""This script trains and tests a given set of Machine Learning algorithms on the various csv files available in  the
data_dir directory. Each csv file has the following columns:[id	target_api , status,siret,categorie_juridique,
categorie_juridique_label,activite_principale,activite_principale_label,nom_raison_sociale,
fondement_juridique_title,fondement_juridique_url,intitule,description,instruction_comment ].

The goal is to predict the 'status' column.

Parameters - in the config/mlapi_parameters.json config file :
1. output_dir: path of the directory where to save all the results from the training and testing of algorithms
2. simple_mode: if set to TRUE, only 3 algorithms are tested (LogisticRegression, XGBoostClassifier,DecisionTreeClassifier)
3. aggregate_cat: if set to TRUE, less frequent categories (<0.4%)in the categorical columns are aggregated
4. (TO DO) explain_mode : if set to TRUE,  XGBoostClassifier,DecisionTreeClassifier are trained and tested
and SHAP plots and a Decision Tree are saved in the output_dir directory

Other parameters - in the script:
1. text_enc: vectorization of text data [TfidfVectorizer(ngram_range=(1, 3)) OR CountVectorizer(ngram_range=(1, 3))]

Script's output:
1. A csv file (*results file* parameter in mlapi_parameters.json) recording all the parameters used in each experience,
identified by an unique ID
2. For each experience, the confusion matrix and the classification report (with all the metrics) in saved in
output_dir/dataset_name_ID
"""

import glob
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import json
from datetime import datetime
import shortuuid
import nltk
from nltk.corpus import stopwords

PARAMETER_FILE = Path("config/mlapi_parameters.json")
if PARAMETER_FILE.exists():
    with open(PARAMETER_FILE) as fout:
        PARAMETERS = json.load(fout)
else:
    raise FileNotFoundError(f"Config file {PARAMETER_FILE.as_posix()} does not exist.")

TEXT_ENC = TfidfVectorizer(ngram_range=(1, 3))


def prepare_results_csv(new_results_row, param):
    """This function fills the new_results_row dictionary with some basic info about the experiment (experience ID, time etc.).
    The dictionary will then be used to add a new line in the results csv file
    :param:     :new_results_row : info about current experience -- dictionary
    :param:     :param : parameters chosen in mlapi_parameters.json"""
    id_ = shortuuid.uuid()
    new_results_row["id"] = id_
    new_results_row["test_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    new_results_row["test_author"] = param["test_author"]
    return new_results_row, id_


def update_results(new_results_row, name_output, data, param):
    """This function updates new_results_row with more info about the dataset: dataset_name,
    dataset_length,refused (%),aggregate_cat,explain_mode. new_results_row will then be used to add a new line
    in the results csv file
    :param:     :new_results_row : info about current experience -- dictionary
    :param:     :name_output: dataset_name -- string
    :param:     :data: dataset -- Pandas Dataframe
    :param:     :param : parameters chosen in mlapi_parameters.json """
    new_results_row["dataset_name"] = name_output
    new_results_row["dataset_length"] = len(data)
    new_results_row["refused (%)"] = round(data.groupby(['status']).count()['id']['refused'] / len(data) * 100,
                                           2)
    new_results_row["aggregate_cat"] = str(param["aggregate_cat"])
    new_results_row["explain_mode"] = str(param["explain_mode"])

    return new_results_row


def drop_useless_var(data):
    """This function drops the useless columns from the data Pandas dataframe."""
    data = data.drop(
        columns=['id', 'siret', 'categorie_juridique', 'activite_principale', 'instruction_comment',
                 'fondement_juridique_url'])
    return data


def impute_nans(data):
    """This function imputes missing values in each column containing missing values by creating a new category
    called *missing*.
    :param:     :data: dataset to impute
    :type:      :data: Pandas dataframe"""
    missing_values_imp = SimpleImputer(strategy='constant', fill_value='missing')
    missing_cols = ['categorie_juridique_label', 'activite_principale_label', 'nom_raison_sociale', 'description',
                    'intitule', 'fondement_juridique_title']
    data[missing_cols] = missing_values_imp.fit_transform(data[missing_cols])
    return data


def aggregate_cat(data, cat_variables):
    """This function aggregates the least common categories (<0.4%) to prepare variables to OneHotEncoding.
    :param:     :data: dataset -- Pandas dataframe
    :param:     :cat_variables: categorical variables -- list"""
    for variable in cat_variables:
        if data[variable].nunique() > 20:
            for category in data[variable].unique():
                if data[variable].value_counts()[category] / len(data) <= 0.004:
                    data.loc[data[variable] == category, variable] = 'Autre'
    return data


def encode_vars(cat_variables, text_enc, new_results_row):
    """This function returns a column transformer for text data and categorical variables.
    :param:     :cat_variables: categorical variables -- list
    :param:     :text_enc: chosen text vectorizer [TfidfVectorizer(ngram_range=(1, 3)) OR CountVectorizer(ngram_range=(1, 3))]
    :param:     :new_results_row : info about current experience -- dictionary
    """
    one_hot_enc = OneHotEncoder(handle_unknown='ignore')
    new_results_row["text_enc"] = str(text_enc)
    columns_trans = make_column_transformer((one_hot_enc, cat_variables), (text_enc, 'nom_raison_sociale'),
                                            (text_enc, 'intitule'),
                                            (text_enc, 'fondement_juridique_title'),
                                            (text_enc, 'description'))
    return columns_trans, new_results_row


def split_train_test(data):
    """This function splits the given dataframe in train and test dataset (75%/25%) with stratification,
    after encoding the target variable. """
    y = data['status'].values
    label_enc = preprocessing.LabelEncoder()
    y = label_enc.fit_transform(y)
    X = data.drop(columns=['status'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def compute_metrics(y_test, prediction, output_dir, name_output, id_, new_results_row, algo_name, parameters_used, results,
                    results_csv):
    """This function computes all the metrics associated with the prediction and saves the output results in
    output_dir.
    :param:     :y_test: scikit-learn y_true -- numpy array
    :param:     :prediction: scikit-learn y_pred -- numpy array
    :param      :output_dir: output direcoty -- Path
    :param:     :id_: experience's ID -- str
    :param:     :new_results_row:  info about current experience -- dictionary
    :param:     :algo_name: name of current scikit-learn algorithm -- str
    :param:     :grid: GridSearchResult
    :param:     :results: updated info about current experience -- Pandas Dataframe
    :param:     :results_csv: Path of the results csv file"""
    report = classification_report(y_test, prediction, output_dict=True)
    pd.DataFrame(report).to_csv(f'{output_dir}/{name_output}_{id_}/classif_report_{algo_name}.csv')
    new_results_row["accuracy"] = report["accuracy"]
    new_results_row["recall_0"] = report["0"]["recall"]
    new_results_row["recall_1"] = report["1"]["recall"]
    new_results_row["precision_0"] = report["0"]["precision"]
    new_results_row["precision_1"] = report["1"]["precision"]
    new_results_row["f_score_macro"] = report["macro avg"]["f1-score"]
    new_results_row["params"] = parameters_used
    confusion = confusion_matrix(y_test, prediction)
    sns.heatmap(confusion, annot=True, vmin=0, vmax=len(y_test), cmap='Blues', fmt='g')
    plt.savefig(f'{output_dir}/{name_output}_{id_}/confusion_matrix_{algo_name}.png')
    plt.close()
    print(f"Output saved in {output_dir}/{name_output}_{id_}")
    results = results.append(new_results_row, ignore_index=True)
    results.to_csv(results_csv, index=False)
    return results


def remove_stopwords(data, text_col):
    """This function removes all the stopwords from the text columns of a given dataframe."""
    stopwords_list = stopwords.words('french')
    for word in ['de la', 'DE', 'DES']:
        stopwords_list.append(word)
    pat = r'\b(?:{})\b'.format('|'.join(stopwords_list))
    for col in text_col:
        data[col] = data[col].str.replace(pat, '')
    return data


def main():
    for param in PARAMETERS:
        data_dir = Path(param["data_dir"])
        output_dir = Path(param["output_dir"])
        results_csv = Path(param["results_file"])
        results = pd.read_csv(results_csv)
        new_results_row = {}
        new_results_row, id_ = prepare_results_csv(new_results_row, param)
        list_csvs = [Path(p) for p in glob.glob(data_dir.joinpath("./*.csv").as_posix())]
        for dataset in list_csvs:
            name_output = dataset.stem
            print(f"Now treating dataset named {name_output}")
            # 1. Read data, drop useless columns and create a proper output folder
            if not output_dir.joinpath(f"{name_output}_{id_}").exists():
                output_dir.joinpath(f"{name_output}_{id_}").mkdir()
            data = pd.read_csv(dataset)
            new_results_row = update_results(new_results_row, name_output, data, param)
            data = drop_useless_var(data)
            # 2. Missing values imputation: create a new category for missing values
            data = impute_nans(data)
            # 3. Aggregate categorical variables because of high cardinality if aggregate_cat param is TRUE
            cat_variables = ['target_api', 'categorie_juridique_label', 'activite_principale_label']
            if param["aggregate_cat"]:
                data = aggregate_cat(data, cat_variables)
            # 4. Encoders (categorical variables & text)
            text_enc = TfidfVectorizer(ngram_range=(1, 3))
            text_col = ['nom_raison_sociale','intitule','fondement_juridique_title','description']
            data = remove_stopwords(data, text_col)
            columns_trans, new_results_row = encode_vars(cat_variables, text_enc, new_results_row)
            # 5. Train/test splitting
            X_train, X_test, y_train, y_test = split_train_test(data)
            # 6. Train and test algorithms
            if param["simple_mode"]:
                algorithms = [LogisticRegression(), RandomForestClassifier(), XGBClassifier()]
                algorithms_names = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier']
            else:
                algorithms = [LogisticRegression(), RandomForestClassifier(), XGBClassifier(), CatBoostClassifier(),
                              SVC(), DecisionTreeClassifier()]
                algorithms_names = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier',
                                    'CatBoostClassifier', 'SVC', 'DecisionTreeClassifier']
            for algorithm, algo_name in zip(algorithms, algorithms_names):
                new_results_row["algo_name"] = algo_name
                print(f"Now starting to fit algorithm: {algo_name}")
                # 7. GridSearch on pipeline
                pipe = make_pipeline(columns_trans, algorithm)
                if param["grid_search"]:
                    param_grid = algorithms_grid[algo_name]
                    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
                    grid.fit(X_train, y_train)
                    print(f"Best estimator:\n{grid.best_estimator_}")
                    prediction = grid.predict(X_test)
                    new_results_row["params"] = str(grid.best_estimator_[algo_name.lower()])
                    parameters_used = str(grid.best_estimator_[algo_name.lower()])
                else:
                    pipe.fit(X_train, y_train)
                    prediction = pipe.predict(X_test)
                    new_results_row["params"] = str(algorithm.get_params())
                    parameters_used = algorithm.get_params(False)
                compute_metrics(y_test, prediction, output_dir, name_output, id_, new_results_row, algo_name,
                                parameters_used, results, results_csv)


algorithms_grid = {'LogisticRegression': {"logisticregression__C": np.arange(0.4, 1.5, 0.2),
                                          "logisticregression__class_weight": ['balanced', {0: .3, 1: .7},
                                                                               {0: .4, 1: .6}, 'auto'],
                                          },
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
                   'DecisionTreeClassifier': {
                       "decisiontreeclassifier__criterion": ["gini", "entropy"],
                       "decisiontreeclassifier__max_depth": np.arange(2, 30, 1),
                   }}

if __name__ == "__main__":
    main()