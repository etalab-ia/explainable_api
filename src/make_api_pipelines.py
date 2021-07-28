"""This script trains and tests a given set of Machine Learning algorithms on the various csv files available in  the
data_dir directory.
The goal is to predict the 'status' column.
Parameters - in the config/mlapi_parameters.json config file :
1. output_dir: path of the directory where to save all the results from the training and testing of algorithms
2. simple_mode: if set to TRUE, only 3 algorithms are tested (LogisticRegression, XGBoostClassifier,DecisionTreeClassifier)
3. aggregate_cat: if set to TRUE, less frequent categories (<0.4%)in the categorical columns are aggregated
4. explain_mode : if set to TRUE,  XGBoostClassifier,DecisionTreeClassifier are trained and tested
and SHAP plots, linear coefficients plots and a Decision Tree are saved in the output_dir directory
5. grid_search: if set to TRUE, a GridSearch is performed using the hyperparameters at the end of the script. Otherwise,
default values are used
Other parameters - in the script:
1. text_enc: vectorization of text data [TfidfVectorizer(ngram_range=(1, 3)) OR CountVectorizer(ngram_range=(1, 3))]
Script's output:
1. A csv file (*results file* parameter in mlapi_parameters.json) recording all the parameters used in each experience,
identified by an unique ID
2. For each experience, the confusion matrix and the classification report (with all the metrics) in saved in
output_dir/dataset_name_ID
"""

# TO DO: fix issue with saving decision tree figure

import glob
import pickle
from collections import defaultdict

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
from nltk.corpus import stopwords
import shap
from sklearn import tree
import mglearn
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import argparse

from src.config.parameter_grid import algorithms_grid


def read_parameters(parameters_file: Path):
    if parameters_file.exists():
        with open(parameters_file) as fout:
            parameters = json.load(fout)
    else:
        raise FileNotFoundError(f"Config file {parameters_file.as_posix()} does not exist.")
    return parameters


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
    new_results_row["explain_mode"] = str(param["explain_mode"])
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


def compute_metrics(y_test, prediction, output_dir, name_output, id_, new_results_row, algo_name, parameters_used):
    """This function computes all the metrics associated with the prediction and saves the output results in
    output_dir.
    """
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

    return new_results_row


def remove_stopwords(data, text_col):
    """This function removes all the stopwords from the text columns of a given dataframe."""
    stopwords_list = stopwords.words('french')
    pattern = r'\b(?:{})\b'.format('|'.join(stopwords_list))
    for col in text_col:
        data[col] = data[col].str.lower()
        data[col] = data[col].str.replace(pattern, '')
    return data


def plot_feature_imp(transformer, model, feature_names, X_train, output_dir, name_output, id_):
    """This function produces SHAP summary plot for feature importance for an XGBoost model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformer.fit_transform(X_train))
    if isinstance(X_train, np.ndarray):
        shap.summary_plot(shap_values, transformer.fit_transform(X_train).toarray(), feature_names=feature_names,
                          show=False, plot_size=(25, 15))
    else:
        shap.summary_plot(shap_values, transformer.fit_transform(X_train), feature_names=feature_names,
                          show=False, plot_size=(25, 15))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name_output}_{id_}/shap_summary_plot.png')
    plt.close()


def plot_shap(transformer, model, feature_names, X_train, output_dir, name_output, id_):
    """This function produces SHAP summary plot for feature importance for an XGBoost model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformer.fit_transform(X_train))
    shap.summary_plot(shap_values, transformer.fit_transform(X_train), plot_type="bar", feature_names=feature_names,
                      show=False, plot_size=(25, 15))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name_output}_{id_}/shap_feature_importance.png')
    plt.close()


def plot_tree(model, feature_names, output_dir, name_output, id_):
    """This function plots the Decision Tree of the DecisionTreeClassifier currently fit."""
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10000 / 100, 12000 / 100), dpi=100)
    tree.plot_tree(model, feature_names=feature_names, max_depth=4, filled=True, fontsize=90, ax=axes)
    plt.savefig(f'{output_dir}/{name_output}_{id_}/decision_tree.png', dpi=100)
    plt.close()


def linear_coefficients(model, feature_names, output_dir, name_output, id_):
    """This function displays the 10 larger coefficients and the 10 smallest coefficients for logistic
    regression"""
    mglearn.tools.visualize_coefficients(
        model.coef_,
        feature_names, n_top_features=10)
    plt.savefig(f'{output_dir}/{name_output}_{id_}/linear_coefficients.png', bbox_inches='tight')
    plt.close()


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


# def display_test(prediction, X_test, y_test, output_dir, name_output, id_, algo_name):
#     """This function creates a csv file of the test dataset together with the result predicted by the algorithm."""
#     list_predicted_status = prediction.tolist()
#     test = X_test.copy()
#     test['real_status'] = y_test
#     test['status_predicted'] = list_predicted_status
#     test_df = pd.DataFrame(test)
#     test_df.to_csv(f'{output_dir}/{name_output}_{id_}/{algo_name}_test_data.csv')


def get_explainer_dashboard(model, X_test, y_test, algo_name,
                            api_name,
                            only_return_explainer=False):
    """Runs the explainerdashboard app according to the model and to the test dataset. The SHAP Explainer (Tree, Linear)
    is chosen according to the algorithm type."""
    if algo_name == 'XGBoostClassifier' or algo_name == 'CatBoostClassifier' or algo_name == 'RandomForestClassifier':
        shap = 'tree'
    elif algo_name == 'LogisticRegression':
        shap = 'linear'
    else:
        shap = 'guess'

    explainer = ClassifierExplainer(model, X_test, y_test, shap=shap)
    dashboard = ExplainerDashboard(explainer, title=f"{api_name}")
    if only_return_explainer:
        return dashboard
    else:
        dashboard.run()


def choose_algo(data):
    """This function returns the best performing algorithm for a given dataset data according to target_api """
    if data['target_api'].str.contains('api_particulier').any():
        algorithms = [CatBoostClassifier()]
        algorithms_names = ['CatBoostClassifier']
    elif data['target_api'].str.contains('api_r2p_sandbox').any():
        algorithms = [CatBoostClassifier()]
        algorithms_names = ['CatBoostClassifier']
    elif data['target_api'].str.contains('franceconnect').any():
        algorithms = [LogisticRegression()]
        algorithms_names = ['LogisticRegression']
    elif data['target_api'].str.contains('aidants_connect').any():
        algorithms = [XGBClassifier()]
        algorithms_names = ['XGBClassifier']
    elif data['target_api'].str.contains('api_entreprise').any():
        algorithms = [LogisticRegression()]
        algorithms_names = ['LogisticRegression']
    elif data['target_api'].str.contains('api_impot_particulier_fc_sandbox').any():
        algorithms = [XGBClassifier()]
        algorithms_names = ['XGBClassifier']
    elif data['target_api'].str.contains('api_ficoba_sandbox').any():
        algorithms = [LogisticRegression()]
        algorithms_names = ['LogisticRegression']
    elif data['target_api'].str.contains('cartobio').any():
        algorithms = [LogisticRegression()]
        algorithms_names = ['LogisticRegression']
    elif data['target_api'].str.contains('api_impot_particulier_sandbox').any():
        algorithms = [RandomForestClassifier()]
        algorithms_names = ['RandomForestClassifier']
    else:
        algorithms = [XGBClassifier()]
        algorithms_names = ['XGBClassifier']
    return algorithms, algorithms_names


def generate_csvs(main_dataset):
    """This function takes the main_dataset csv file and separates it in multiple csv files according to the target_api type.
    """
    main_df = pd.read_csv(main_dataset)
    for api in main_df['target_api'].unique():
        api_df = main_df[main_df['target_api'] == api]
        if len(api_df) > 20:
            api_df.to_csv(f'./data/data_by_api/output_{api}.csv', index_label=None)


def main(parameters_file: Path):
    parameters = read_parameters(parameters_file=parameters_file)
    for param in parameters:
        data_dir = Path(param["data_dir"])
        output_dir = Path(param["output_dir"])
        results_csv = Path(param["results_file"])
        per_api_data_pkl = Path(param["per_api_data_file"])

        results = pd.read_csv(results_csv)

        new_results_row = {}
        new_results_row, id_ = prepare_results_csv(new_results_row, param)
        # generate csvs according to target_api
        main_dataset = glob.glob(data_dir.joinpath("./*.csv").as_posix())[0]
        generate_csvs(main_dataset=main_dataset)
        # list of csv to be treated
        list_csvs = [Path(p) for p in glob.glob(data_dir.joinpath("./data_by_api/*.csv").as_posix())]
        dict_api_expe = defaultdict(dict)
        expe_info = {}

        for dataset in list_csvs[:]:
            name_output = dataset.stem
            try:

                print(f"Now treating dataset named {name_output}")
                # 1. Create an output folder
                if not output_dir.joinpath(f"{name_output}_{id_}").exists():
                    output_dir.joinpath(f"{name_output}_{id_}").mkdir()
                # 2. Read data
                data = pd.read_csv(dataset)
                # 3. Preprocess data for ML
                text_enc = TfidfVectorizer(ngram_range=(1, 3))
                new_results_row = update_results(new_results_row, name_output, data, param, text_enc)
                agg_cat = param["aggregate_cat"]
                X, y, columns_trans = preprocess_data(data, text_enc, agg_cat, features=FEATURES)
                # 4. Train/test splitting
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                # 5. Train and test algorithms
                if param['explainerdashboard']:
                    algorithms, algorithms_names = choose_algo(data)
                else:
                    if param["simple_mode"]:
                        algorithms = [LogisticRegression(), RandomForestClassifier(), XGBClassifier()]
                        algorithms_names = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier']
                    else:
                        algorithms = [LogisticRegression(), RandomForestClassifier(), XGBClassifier(),
                                      CatBoostClassifier(),
                                      SVC(), DecisionTreeClassifier()]
                        algorithms_names = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier',
                                            'CatBoostClassifier', 'SVC', 'DecisionTreeClassifier']
                for algorithm, algo_name in zip(algorithms, algorithms_names):
                    new_results_row["algo_name"] = algo_name
                    print(f"Now starting to fit algorithm: {algo_name}")
                    # 6. GridSearch on pipeline
                    pipe = make_pipeline(columns_trans, algorithm)
                    if param["grid_search"]:
                        param_grid = algorithms_grid[algo_name]
                        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
                        grid.fit(X_train, y_train)
                        transformer, feature_names, model = info_from_pipeline(grid, param, algo_name=algo_name.lower())
                        # 7. SHAP plot
                        if param["explain_mode"]:
                            if algo_name == 'XGBClassifier':
                                plot_feature_imp(transformer, model, feature_names, X_train, output_dir, name_output,
                                                 id_)
                                plot_shap(transformer, model, feature_names, X_train, output_dir, name_output, id_)
                            elif algo_name == 'DecisionTreeClassifier':
                                plot_tree(model, feature_names, output_dir, name_output, id_)
                            elif algo_name == 'LogisticRegression':
                                linear_coefficients(model, feature_names, output_dir, name_output, id_)
                        print(f"Best estimator:\n{grid.best_estimator_}")
                        prediction = grid.predict(X_test)
                        parameters_used = str(grid.best_estimator_[algo_name.lower()])
                        new_results_row = compute_metrics(y_test, prediction, output_dir, name_output, id_,
                                                          new_results_row,
                                                          algo_name,
                                                          parameters_used)
                        results = results.append(new_results_row, ignore_index=True)
                        results.to_csv(results_csv, index=False)
                        expe_info = {"model": grid.best_estimator_,
                                     "X_test": X_test, "y_test": y_test, "algo_name": algo_name}
                    else:
                        pipe.fit(X_train, y_train)
                        transformer, feature_names, model = info_from_pipeline(pipe, param, algo_name=algo_name.lower())
                        if param["explain_mode"]:
                            if algo_name == 'XGBClassifier':
                                plot_feature_imp(transformer, model, feature_names, X_train, output_dir, name_output,
                                                 id_)
                                plot_shap(transformer, model, feature_names, X_train, output_dir, name_output, id_)
                            elif algo_name == 'DecisionTreeClassifier':
                                plot_tree(model, feature_names, output_dir, name_output, id_)
                            elif algo_name == 'LogisticRegression':
                                linear_coefficients(model, feature_names, output_dir, name_output, id_)
                        prediction = pipe.predict(X_test)
                        parameters_used = algorithm.get_params(False)
                        new_results_row = compute_metrics(y_test, prediction, output_dir, name_output, id_,
                                                          new_results_row,
                                                          algo_name,
                                                          parameters_used)
                        results = results.append(new_results_row, ignore_index=True)
                        results.to_csv(results_csv, index=False)
                        expe_info = {"model": pipe, "X_test": X_test, "y_test": y_test, "algo_name": algo_name}
                dict_api_expe[name_output] = expe_info
            except Exception as e:
                print(f"Could not experiment with dataset {name_output}. Error: {e}")

        print(f"Saving experiments per API to {per_api_data_pkl}")
        with open(per_api_data_pkl, "wb") as filo:
            pickle.dump(dict_api_expe, filo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="./src/config/mlapi_parameters.json")
    args = parser.parse_args()
    main(Path(args.config_file))
