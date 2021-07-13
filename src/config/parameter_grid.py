import numpy as np

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