import pandas as pd
# import numpy as np
# import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier as rf


dataset = pd.read_csv('csv_result-ebay_confianca_completo.csv')
dataset['blacklist'] = dataset['blacklist'] == 'S'

X = dataset.iloc[:, 0:74].values
y = dataset.iloc[:, 74].values


# Treinamento com RandomForest ou GaussianNB


X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size=0.2,
                                                                  stratify = y)

model = rf()
model.fit(X_treinamento, y_treinamento)
previsao = model.predict(X_teste)

accuracy_score(y_teste, previsao)


# Treinamento com Subamostragem


tl = TomekLinks(sampling_strategy='majority')
X_sub, y_sub = tl.fit_resample(X, y)

X_treinamento_s, X_teste_s, y_treinamento_s, y_teste_s = train_test_split(X_sub, y_sub,
                                                                          test_size=0.2,
                                                                          stratify = y_sub)

model_s = rf()
model_s.fit(X_treinamento_s, y_treinamento_s)
previsao_s = model_s.predict(X_teste_s)

accuracy_score(y_teste_s, previsao_s)


# Treinamento com Sobreamostragem


smote = SMOTE(sampling_strategy='minority')
X_sob, y_sob = smote.fit_resample(X,y)

X_treinamento_so, X_teste_so, y_treinamento_so, y_teste_so = train_test_split(X_sob, y_sob,
                                                                              test_size=0.2,
                                                                              stratify = y_sob)

model_so = rf()
model_so.fit(X_treinamento_so, y_treinamento_so)
previsao_so = model_so.predict(X_teste_so)

accuracy_score(y_teste_so, previsao_so)
