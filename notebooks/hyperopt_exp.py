# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import mlflow

# %config InlineBackend.figure_format='retina'

# %load_ext dotenv
# %dotenv

# %%
train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

X_train = train_df['text']
y_train = train_df['humor'] 

# %%
search_space = {
    'preprocessing': hp.choice(
        'preprocessing_type',
        [
            {
                'preprocessing_type': 'count_vec',
                'max_df': hp.uniform('count_max_df', 0.2, 1.0),
                'min_df': hp.uniform('count_min_df', 0.0, 0.1),
                'max_features': hp.choice('count_max_features', np.arange(100, 5000, dtype=int))
            },
            {
                'preprocessing_type': 'tfidf_vec',
                'max_df': hp.uniform('tfidf_max_df', 0.2, 1.0),
                'min_df': hp.uniform('tfidf_min_df', 0.0, 0.1),
                'max_features': hp.choice('tfidf_max_features', np.arange(100, 5000, dtype=int)),
                'use_idf': hp.choice('tfidf_use_idf', [False, True])
            }
        ]
    ),
        'model': hp.choice(
        'classifier_type',
        [
            {
                'model_type': 'rf',
                'n_estimators': hp.choice('n_estimators', np.arange(30, 300, dtype=int)),
                'max_depth': hp.choice('max_depth', np.arange(1, 10, dtype=int)),
                'criterion': hp.choice('criterion', ['gini', 'entropy'])
            },
            {
                'model_type': 'logreg',
                'C': hp.lognormal('LR_C', 0, 1.0),
                'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
            },
        ]
    )
}


# %%
def objective(params):

    preprocessing_type = params['preprocessing']['preprocessing_type']
    classifier_type = params['model']['model_type']
    
    with mlflow.start_run():
        mlflow.log_params(params['preprocessing'])
        mlflow.log_params(params['model'])
        
        del params['preprocessing']['preprocessing_type']
        
        del params['model']['model_type']
        
        # preprocessing
        if preprocessing_type == 'count_vec':
            preproc = CountVectorizer(**params['preprocessing'])
        elif preprocessing_type == 'tfidf_vec':
            preproc = TfidfVectorizer(**params['preprocessing'])
        else:
            return 0

        # model
        if classifier_type == 'rf':
            clf = RandomForestClassifier(**params['model'])
        elif classifier_type == 'logreg':
            clf = LogisticRegression(**params['model'])
        else:
            return 0
        
        # TODO логировать время

        t0 = time.time()
        X_train_processed = preproc.fit_transform(X_train)
        accuracy = cross_val_score(clf, X_train_processed, y_train, cv=3).mean()
        
        mlflow.log_metric('accuracy', accuracy)
        # print(f'accuracy: {accuracy}')
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}


# %%
mlflow.set_experiment('otus_hyperflow_exp')

# %%
trials = Trials()

best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)
