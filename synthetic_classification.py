#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.utils import to_categorical

from synthetic_generator import generate_data
from feature_selector import FeatureSelector

#%% Define constants 

NUM = 1500
EPOCHS = 100
FACTOR_LOSS = 3
EPOCHS = 400

#%% Prepare dataset. 

df = generate_data(linear=True, classification=True)
X = df.drop(['y'], axis=1)
y = to_categorical(df['y'].to_numpy())

#%%

eighty = int(2/3*NUM)
X_train = X[:eighty]
X_test = X[eighty:]
y_train = y[:eighty]
y_test = y[eighty:]


#%% Perform selection.

selector = FeatureSelector(task_nodes = [10,20,60,30,20,10,2],
               selector_nodes = [10,100,75,50,10],
               task_activation='sigmoid')

selector.compile(factor_loss=FACTOR_LOSS,
            regression = False,
            loss = 'categorical_crossentropy',
            metric = 'categorical_accuracy',
            threhold=0.01) # For non-linear classification change threshold to 0.1

history = selector.fit(EPOCHS, X_train, y_train, validation_data=(X_test, y_test))

#%% Print feature selection. Not that just the feature indices are printed. 

selected_features = selector.eval()
print(f'Selected: {selected_features[0]}. Size: {len(selected_features)}')
