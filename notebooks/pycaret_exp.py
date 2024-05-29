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
import mlflow

import pandas as pd

from nltk.corpus import stopwords

from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model

# %config InlineBackend.figure_format='retina'

# %load_ext dotenv
# %dotenv

# %% [markdown]
# ## Подготовка данных

# %%
#df = pd.read_csv('../data/humor.csv')
#test_df = df.sample(n=1_000)
#train_df = df.drop(test_df.index).sample(n=5_000)
#train_df.to_csv('../data/train_data.csv', index=False)
#test_df.to_csv('../data/test_data.csv', index=False)

# %%
train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')
train_df.head()

# %% [markdown]
# ## Обучение модели

# %%
# Инициализация pycaret

s = setup(
  data=train_df,
  test_data=test_df,
  index=False,
  target='humor',
  log_experiment=True,
  experiment_name='otus_pycaret',
  text_features=['text'],
  text_features_method='tf-idf'
)

# %%
# Сранение моделей
best = compare_models(
  #cross_validation=False, 
  exclude=['qda', 'lda', 'gbc'],
)

print(best)

# %% [markdown]
# ## Тренируем выбранную модель

# %%
model = create_model('lr')

# %%
# Проверяем гиперпараметры

model = tune_model(model, choose_better=True)

# %%
plot_model(model)
