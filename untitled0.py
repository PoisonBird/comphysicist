# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:56:57 2019

@author: rdati
"""

import numpy as np
import pandas as pd

df = pd.read_csv('Dataset.txt',
                 header = 0,
                 index_col = False,
                 sep = "\s+")

train = df.loc[df.Target_validation == 0, :]
test = df.loc[df.Target_validation == 1, :]

X1_term = train.iloc[:30]

X1 = X1_term[['A1','A2']]

y1 = X1_term["Target1"]

from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.1, random_state = 0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X1_train, y1_train)

y1_pred = regressor.predict(X1_test)
y1_pred = pd.DataFrame(y1_pred)
y1_pred.columns = ['Prediction']


from sklearn.metrics import mean_squared_error as mse

print("MSE: ", mse(y1_test, y1_pred ))

y1_test.to_csv("results.csv", mode = 'w')

read = pd.read_csv("results.csv", index_col = False, names = ['ID', 'Target'])
reads = pd.concat([read, y1_pred], axis = 1)

reads.to_csv("results.csv", mode = 'w')

# 여기까지가 모델 성능평가하는 코드..

def model_creator():
    
    a=7
    b=12
    
    X_term = train.iloc[a: b]
    
    X = X_term[['A1', 'A2']]
    
    y= X_term["Target1"]
    
    regressor_x = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor_x.fit(X, y)
    
    X_test = test.loc[b > test.ID, ['A1', 'A2']]
    X_test = X_test.loc[a < test.ID]
    
    pred = regressor_x.predict(X_test)
    pred = pd.DataFrame(pred)
    pred.columns = ['Prediction']
    
    pred.to_csv("Prediction Results.csv", mode = 'a')
    
model_creator()