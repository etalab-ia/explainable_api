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

# TO DO: fix issue with saving decision tree figure

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
from nltk.corpus import stopwords
import shap
from sklearn import tree
import mglearn

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
    new_results_row["all_features"] = str(param["all_features"])
    return new_results_row


def modify_act_principale(data):
    """Convert the codes NAF (https://www.legifrance.gouv.fr/loda/id/JORFTEXT000017765090/) of the activite_principale column
    to meaningful numerical values."""
    data['activite_principale'] = data['activite_principale'].str[:-1]
    data['activite_principale'] = pd.to_numeric(data["activite_principale"], downcast="float")
    return data['activite_principale']


def preprocess_data(data, text_enc, agg_cat, all_features):
    """Returns X and y for a given dataset together with the adequate column transformer for a given text transformer"""
    if all_features:
        data = data.drop(columns=['id', 'siret','instruction_comment',
                                      'fondement_juridique_url'])
        data['activite_principale'] = modify_act_principale(data)
    else:
        data = data.drop(columns=['id', 'siret', 'categorie_juridique', 'activite_principale', 'instruction_comment',
                                  'fondement_juridique_url'])
    data = impute_nans(data, all_features=all_features)
    cat_variables = ['target_api', 'categorie_juridique_label', 'activite_principale_label']
    if agg_cat:
        data = aggregate_cat(data, cat_variables)
    text_col = ['nom_raison_sociale', 'intitule', 'fondement_juridique_title', 'description']
    data = remove_stopwords(data, text_col)
    one_hot_enc = OneHotEncoder(handle_unknown='ignore')
    label_enc = preprocessing.LabelEncoder()
    columns_trans = make_column_transformer((one_hot_enc, cat_variables), (text_enc, 'nom_raison_sociale'),
                                            (text_enc, 'intitule'),
                                            (text_enc, 'fondement_juridique_title'),
                                            (text_enc, 'description'))
    y = data['status'].values
    y = label_enc.fit_transform(y)
    X = data.drop(columns=['status'])
    return X, y, columns_trans


def impute_nans(data, all_features):
    """This function imputes missing values in each column containing missing values by creating a new category
    called *missing*.
    :param:     :data: dataset to impute
    :type:      :data: Pandas dataframe
    :param:     :all_features: whether to consider all meaningful features or not
    :type:      :all_features: bool"""
    missing_values_imp = SimpleImputer(strategy='constant', fill_value='missing')
    missing_cols = ['categorie_juridique_label', 'activite_principale_label', 'nom_raison_sociale', 'description',
                    'intitule', 'fondement_juridique_title']
    data[missing_cols] = missing_values_imp.fit_transform(data[missing_cols])
    if all_features:
        numerical_imputer = SimpleImputer(strategy='constant', fill_value=0)
        numerical_missing = ['categorie_juridique', 'activite_principale']
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
    :param:     :grid: GridSearch results if grid_search parameter is TRUE; Pipeline parameters otherwise
    :param:     :X_train: X_trains set; numpy array
    :param:     :output_dir: name of the output directory where to save shap plot; str
    :param:     :name_output: name of the algorithm; str
    :param:     :id_: experiment's ID; str
    :param:     :param: parameters from mlapi_parameters.json"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformer.fit_transform(X_train))
    shap.summary_plot(shap_values, transformer.fit_transform(X_train).toarray(), feature_names=feature_names,
                      show=False, plot_size=(25, 15))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name_output}_{id_}/shap_summary_plot.png')
    plt.close()


def plot_shap(transformer, model, feature_names, X_train, output_dir, name_output, id_):
    """This function produces SHAP summary plot for feature importance for an XGBoost model.
    :param:     :grid: GridSearch results if grid_search parameter is TRUE; Pipeline parameters otherwise
    :param:     :X_train: X_trains set; numpy array
    :param:     :output_dir: name of the output directory where to save shap plot; str
    :param:     :name_output: name of the algorithm; str
    :param:     :id_: experiment's ID; str
    :param:     :param: parameters from mlapi_parameters.json"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformer.fit_transform(X_train))
    shap.summary_plot(shap_values, transformer.fit_transform(X_train), plot_type="bar", feature_names=feature_names,
                      show=False, plot_size=(25, 15))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name_output}_{id_}/shap_feature_importance.png')
    plt.close()


def plot_tree(model, feature_names, output_dir, name_output, id_):
    """This function plots the Decision Tree of the DecisionTreeClassifier currently fit.
    :param:     :grid: GridSearch results if grid_search parameter is TRUE; Pipeline parameters otherwise
    :param:     :X_train: X_trains set; numpy array
    :param:     :output_dir: name of the output directory where to save shap plot; str
    :param:     :name_output: name of the algorithm; str
    :param:     :id_: experiment's ID; str
    :param:     :param: parameters from mlapi_parameters.json"""
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


def display_test(prediction, X_test, y_test, output_dir, name_output, id_, algo_name):
    """This function returns a csv file of the test dataset together with the result predicted by the algorithm."""
    list_predicted_status = prediction.tolist()
    test = X_test.copy()
    test['real_status'] = y_test
    test['status_predicted'] = list_predicted_status
    test_df = pd.DataFrame(test)
    test_df.to_csv(f'{output_dir}/{name_output}_{id_}/{algo_name}_test_data.csv')


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
            # 1. Create an output folder
            if not output_dir.joinpath(f"{name_output}_{id_}").exists():
                output_dir.joinpath(f"{name_output}_{id_}").mkdir()
            # 2. Read data
            data = pd.read_csv(dataset)
            # 3. Preprocess data for ML
            text_enc = TfidfVectorizer(ngram_range=(1, 3))
            new_results_row = update_results(new_results_row, name_output, data, param, text_enc)
            agg_cat = param["aggregate_cat"]
            X, y, columns_trans = preprocess_data(data, text_enc, agg_cat, all_features=param["all_features"])
            # 4. Train/test splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            # 5. Train and test algorithms
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
                            plot_feature_imp(transformer, model, feature_names, X_train, output_dir, name_output, id_)
                            plot_shap(transformer, model, feature_names, X_train, output_dir, name_output, id_)
                        elif algo_name == 'DecisionTreeClassifier':
                            plot_tree(model, feature_names, output_dir, name_output, id_)
                        elif algo_name == 'LogisticRegression':
                            linear_coefficients(model, feature_names, output_dir, name_output, id_)
                    print(f"Best estimator:\n{grid.best_estimator_}")
                    prediction = grid.predict(X_test)
                    parameters_used = str(grid.best_estimator_[algo_name.lower()])
                    new_results_row = compute_metrics(y_test, prediction, output_dir, name_output, id_, new_results_row,
                                                      algo_name,
                                                      parameters_used)
                    results = results.append(new_results_row, ignore_index=True)
                    results.to_csv(results_csv, index=False)
                    display_test(prediction, X_test, y_test, output_dir, name_output, id_, algo_name)
                else:
                    pipe.fit(X_train, y_train)
                    transformer, feature_names, model = info_from_pipeline(pipe, param, algo_name=algo_name.lower())
                    if param["explain_mode"]:
                        if algo_name == 'XGBClassifier':
                            plot_feature_imp(transformer, model, feature_names, X_train, output_dir, name_output, id_)
                            plot_shap(transformer, model, feature_names, X_train, output_dir, name_output, id_)
                        elif algo_name == 'DecisionTreeClassifier':
                            plot_tree(model, feature_names, output_dir, name_output, id_)
                        elif algo_name == 'LogisticRegression':
                            linear_coefficients(model, feature_names, output_dir, name_output, id_)
                    prediction = pipe.predict(X_test)
                    parameters_used = algorithm.get_params(False)
                    new_results_row = compute_metrics(y_test, prediction, output_dir, name_output, id_, new_results_row,
                                                      algo_name,
                                                      parameters_used)
                    results = results.append(new_results_row, ignore_index=True)
                    results.to_csv(results_csv, index=False)
                    display_test(prediction, X_test, y_test, output_dir, name_output, id_, algo_name)


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
