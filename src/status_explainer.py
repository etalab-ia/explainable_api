"""This script is a shorter and simpler version of the main script. This code only separates the main cvs according to the target_api,
trains and tests the best algorithm accordingly and displays the results through explainerdashboard.
INPUT: csv file and chosen API
#TO DO: impose a minimum size for the df
OUTPUT: explainerdahsboard"""

import glob

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
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pathlib import Path
import json
from datetime import datetime
import shortuuid
from nltk.corpus import stopwords
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

PARAMETER_FILE = Path("config/status_explainer_parameters.json")
if PARAMETER_FILE.exists():
    with open(PARAMETER_FILE) as fout:
        PARAMETERS = json.load(fout)
else:
    raise FileNotFoundError(f"Config file {PARAMETER_FILE.as_posix()} does not exist.")

FEATURES = ['categorie_juridique_label', 'activite_principale_label', 'target_api', 'nom_raison_sociale', 'intitule',
            'fondement_juridique_title', 'description']


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


def update_results(new_results_row, name_output, data, param, text_enc):
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
    new_results_row["text_enc"] = str(text_enc)
    new_results_row["grid_search"] = str(param["grid_search"])
    new_results_row["features"] = FEATURES
    return new_results_row


def modify_act_principale(data):
    """Convert the codes NAF (https://www.legifrance.gouv.fr/loda/id/JORFTEXT000017765090/) of the activite_principale column
    to meaningful numerical values."""
    data['activite_principale'] = data['activite_principale'].str[:-1]
    data['activite_principale'] = pd.to_numeric(data["activite_principale"], downcast="float")
    return data['activite_principale']


def preprocess_data(data, text_enc, agg_cat, features):
    """Returns X and y for a given dataset together with the adequate column transformer for a given text transformer"""
    if len(features) == 0:
        raise ValueError('Please give at least one feature.')
    else:
        useless_col = ['id', 'siret', 'instruction_comment',
                       'fondement_juridique_url']
        data = data.drop(columns=useless_col)
        cat_variables = ['target_api', 'categorie_juridique_label', 'activite_principale_label']
        cat_to_encode = []
        for feature in features:
            if feature in cat_variables:
                cat_to_encode.append(feature)
            elif feature == 'activite_principale':
                data['activite_principale'] = modify_act_principale(data)
            elif feature == 'status':
                raise ValueError('The status column in the target variable and cannot be considered as feature.'
                                 'Please remove status from FEATURES list')
        data = impute_nans(data, features=features)
        if agg_cat:
            data = aggregate_cat(data, cat_variables=cat_to_encode)
        text_col = ['nom_raison_sociale', 'intitule', 'fondement_juridique_title', 'description']
        one_hot_enc = OneHotEncoder(handle_unknown='ignore')
        if len(cat_to_encode) != 0:
            if set(text_col).issubset(features):
                data = remove_stopwords(data, text_col)
                columns_trans = make_column_transformer((one_hot_enc, cat_to_encode), (text_enc, 'nom_raison_sociale'),
                                                        (text_enc, 'intitule'),
                                                        (text_enc, 'fondement_juridique_title'),
                                                        (text_enc, 'description'))
            else:
                columns_trans = make_column_transformer((one_hot_enc, cat_to_encode))
        label_enc = preprocessing.LabelEncoder()
        y = data['status'].values
        y = label_enc.fit_transform(y)
        X = data[features]
    return X, y, columns_trans


def impute_nans(data, features):
    """This function imputes missing values in each column containing missing values by creating a new category
    called *missing*."""
    missing_values_imp = SimpleImputer(strategy='constant', fill_value='missing')
    missing_cols = ['categorie_juridique_label', 'activite_principale_label', 'nom_raison_sociale', 'description',
                    'intitule', 'fondement_juridique_title']
    missing_cols = [feature for feature in features if feature in missing_cols]
    if set(missing_cols).issubset(features) and len(missing_cols) != 0:
        data[missing_cols] = missing_values_imp.fit_transform(data[missing_cols])
    numerical_imputer = SimpleImputer(strategy='constant', fill_value=0)
    numerical_missing = ['categorie_juridique', 'activite_principale']
    numerical_missing = [feature for feature in features if feature in numerical_missing]
    if set(numerical_missing).issubset(features) and len(numerical_missing) != 0:
        data[numerical_missing] = numerical_imputer.fit_transform(data[numerical_missing])
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


def compute_metrics(y_test, prediction, new_results_row, parameters_used):
    """This function computes all the metrics associated with the prediction and saves the output results in
    output_dir.
    """
    report = classification_report(y_test, prediction, output_dict=True)
    new_results_row["accuracy"] = report["accuracy"]
    new_results_row["recall_0"] = report["0"]["recall"]
    new_results_row["recall_1"] = report["1"]["recall"]
    new_results_row["precision_0"] = report["0"]["precision"]
    new_results_row["precision_1"] = report["1"]["precision"]
    new_results_row["f_score_macro"] = report["macro avg"]["f1-score"]
    new_results_row["params"] = parameters_used
    return new_results_row


def remove_stopwords(data, text_col):
    """This function removes all the stopwords from the text columns of a given dataframe."""
    stopwords_list = stopwords.words('french')
    pattern = r'\b(?:{})\b'.format('|'.join(stopwords_list))
    for col in text_col:
        data[col] = data[col].str.lower()
        data[col] = data[col].str.replace(pattern, '')
    return data


def info_from_pipeline(grid, param, algo_name):
    """This function extracts some useful info from the pipeline: the model, the columns transformer and the
    features names"""
    if param['grid_search']:
        transformer = grid.best_estimator_.named_steps['columntransformer']
        model = grid.best_estimator_.named_steps[algo_name]
    else:
        transformer = grid.named_steps['columntransformer']
        model = grid.named_steps[algo_name]
    feature_names = transformer.get_feature_names()
    return transformer, feature_names, model


def explainer_dashboard(model, X_test, y_test, algo_name):
    """Runs the explainerdashboard app according to the model and to the test dataset. The SHAP Explainer (Tree, Linear)
    is chosen according to the algorithm type."""
    if algo_name == 'XGBoostClassifier' or algo_name == 'CatBoostClassifier' or algo_name == 'RandomForestClassifier':
        shap = 'tree'
    elif algo_name == 'LogisticRegression':
        shap = 'linear'
    else:
        shap = 'guess'
    explainer = ClassifierExplainer(model, X_test, y_test, shap=shap)
    ExplainerDashboard(explainer).run()


def choose_algo(data):
    """This function returns the best performing algorithm for a given dataset data according to target_api """
    if data['target_api'].str.contains('api_particulier').any():
        algorithms = CatBoostClassifier()
        algorithms_names = 'CatBoostClassifier'
    elif data['target_api'].str.contains('api_r2p_sandbox').any():
        algorithms = CatBoostClassifier()
        algorithms_names = 'CatBoostClassifier'
    elif data['target_api'].str.contains('franceconnect').any():
        algorithms = LogisticRegression()
        algorithms_names = 'LogisticRegression'
    elif data['target_api'].str.contains('aidants_connect').any():
        algorithms = XGBClassifier()
        algorithms_names = 'XGBClassifier'
    elif data['target_api'].str.contains('api_entreprise').any():
        algorithms = LogisticRegression()
        algorithms_names = 'LogisticRegression'
    elif data['target_api'].str.contains('api_impot_particulier_fc_sandbox').any():
        algorithms = XGBClassifier()
        algorithms_names = 'XGBClassifier'
    elif data['target_api'].str.contains('api_ficoba_sandbox').any():
        algorithms = LogisticRegression()
        algorithms_names = 'LogisticRegression'
    elif data['target_api'].str.contains('cartobio').any():
        algorithms = LogisticRegression()
        algorithms_names = 'LogisticRegression'
    elif data['target_api'].str.contains('api_impot_particulier_sandbox').any():
        algorithms = RandomForestClassifier()
        algorithms_names = 'RandomForestClassifier'
    else:
        algorithms = XGBClassifier()
        algorithms_names = 'XGBClassifier'
    return algorithms, algorithms_names


def main():
    for param in PARAMETERS:
        data_path = Path(param["data_path"])
        results_csv = Path(param["results_file"])
        results = pd.read_csv(results_csv)
        new_results_row = {}
        new_results_row, id_ = prepare_results_csv(new_results_row, param)
        name_api = param['target_api']
        # 1. Read main csv file
        main_data = pd.read_csv(data_path)
        # 2. Subset of main df according to target API
        data = main_data[main_data['target_api']==name_api]
        # 3. Preprocess data for ML
        text_enc = TfidfVectorizer(ngram_range=(1, 3))
        new_results_row = update_results(new_results_row, name_api, data, param, text_enc)
        agg_cat = param["aggregate_cat"]
        X, y, columns_trans = preprocess_data(data, text_enc, agg_cat, features=FEATURES)
        # 3. Train/test splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        # 4. Train and test algorithms
        algorithm, algo_name = choose_algo(data)
        new_results_row["algo_name"] = algo_name
        print(f"Now starting to fit algorithm: {algo_name}")
        # 5. GridSearch on pipeline
        pipe = make_pipeline(columns_trans, algorithm)
        if param["grid_search"]:
            param_grid = algorithms_grid[algo_name]
            grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
            grid.fit(X_train, y_train)
            prediction = grid.predict(X_test)
            parameters_used = str(grid.best_estimator_[algo_name.lower()])
            new_results_row = compute_metrics(y_test, prediction, new_results_row, parameters_used)
            results = results.append(new_results_row, ignore_index=True)
            results.to_csv(results_csv, index=False)
            explainer_dashboard(model=grid, X_test=X_test, y_test=y_test, algo_name=algo_name)
        else:
            pipe.fit(X_train, y_train)
            prediction = pipe.predict(X_test)
            parameters_used = algorithm.get_params(False)
            new_results_row = compute_metrics(y_test, prediction, new_results_row, parameters_used)
            results = results.append(new_results_row, ignore_index=True)
            results.to_csv(results_csv, index=False)
            explainer_dashboard(model=pipe, X_test=X_test, y_test=y_test, algo_name=algo_name)


algorithms_grid = {'LogisticRegression': {"logisticregression__C": np.arange(0.4, 1.5, 0.2),
                                          "logisticregression__class_weight": ['balanced', {0: .3, 1: .7},
                                                                               {0: .4, 1: .6}, 'auto']
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
                       "xgbclassifier__subsample": [0.5, 1.0]},
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
