#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys 

sys.path.append('C:/Users/abbassi/Desktop/DNFS')

from feature_selector import FeatureSelector

#%% Define constants 

# Add path here
# LOAD_PATH = sys.argv[1]
LOAD_PATH = 'C:/Users/abbassi/Desktop/DNFS'

factor_loss = 20
epochs = 200

#%% Prepare dataset. 

df = pd.read_csv(f'{LOAD_PATH}/AmesHousing.csv')

df = df.fillna(0)

X = df.drop(['SalePrice', 'PID', 'Order'], axis=1)
X = pd.get_dummies(X, drop_first=True)

y = df['SalePrice']
y_raw = df['SalePrice']

y_min = y_raw.min()  # y_raw should be the original unnormalized target values
y_max = y_raw.max()

# Normalize target data
y = (y-y.min())/(y.max()-y.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

std_scaler = preprocessing.StandardScaler()
X_train = std_scaler.fit_transform(X_train.astype(float))
X_test = std_scaler.transform(X_test.astype(float))


#%% Perform selection.

selector = FeatureSelector(selector_nodes = [276,256,276],
               task_nodes = [276, 128, 64, 32, 1])

selector.compile(factor_loss = factor_loss)

history = selector.fit(epochs, X_train, y_train, validation_data=(X_test, y_test), verbose=2)

#%% Print feature selection. Not that just the feature indices are printed. 

selected_features = selector.eval()
print(f'Selected: {selected_features}. Size: {len(selected_features)}')

#%% Evaluate with SVM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

selector.build_SVM(is_regression=True)
selector.SVM.fit(X_train, y_train)

# Predict normalized outputs
y_train_pred_norm = selector.SVM.predict(X_train)
y_test_pred_norm = selector.SVM.predict(X_test)

# Inverse transform predictions and ground truth to original scale
y_train_pred = y_train_pred_norm * (y_max - y_min) + y_min
y_test_pred = y_test_pred_norm * (y_max - y_min) + y_min
y_train_orig = y_train * (y_max - y_min) + y_min
y_test_orig = y_test * (y_max - y_min) + y_min

# Compute MAE in original dollar units
mae_train = mean_absolute_error(y_train_orig, y_train_pred)
mae_test = mean_absolute_error(y_test_orig, y_test_pred)


print(f"MAE train: {mae_train:.4f}")
print(f"MAE test: {mae_test:.4f}")
