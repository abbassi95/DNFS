#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from synthetic_generator import generate_data
from feature_selector import FeatureSelector

#%% Define constants 

NUM = 1500
EPOCHS = 100
FACTOR_LOSS = 2
EPOCHS = 400

#%% Prepare dataset. 

df = generate_data(linear=True)
X = df.drop(['y'], axis=1)
y = df['y']

eighty = int(2/3*NUM)
X_train = X[:eighty]
X_test = X[eighty:]
y_train = y[:eighty]
y_test = y[eighty:]

#%% Perform selection.

selector = FeatureSelector(task_nodes = [10,20,100,50,50,25,1],
               selector_nodes = [10,100,75,50,10],
               task_activation='sigmoid')

selector.compile(factor_loss=FACTOR_LOSS)


history = selector.fit(EPOCHS, X_train, y_train, validation_data=(X_test, y_test))

#%% Print feature selection. Not that just the feature indices are printed. 

selected_features = selector.eval()
print(f'Selected: {selected_features[0]}. Size: {len(selected_features)}')