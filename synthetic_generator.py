import pandas as pd
import numpy as np
from tensorflow import keras
from keras import Model


def generate_data(linear=True, classification=False, size=1500, seed=40):
    columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'y']
    df = pd.DataFrame(columns=columns)
    np.random.seed(seed)
    df['x0'] = np.random.standard_normal(size=(size,))
    df['x1'] = np.random.standard_normal(size=(size,))
    df['x2'] = np.random.standard_normal(size=(size,))
    df['x3'] = np.random.standard_normal(size=(size,))
    df['x4'] = np.random.standard_normal(size=(size,))
    df['x9'] = np.random.standard_normal(size=(size,))
    if linear:
        df['x5'] = 6*df['x2']
        df['x6'] = -.5*df['x3']
        df['x7'] = 5*df['x0']
        df['x8'] = -2*df['x1']
    else:
        df['x5'] = df['x2']**3
        df['x6'] = -.5*df['x3']**5/(1+np.exp(df['x0']))
        df['x7'] = 5*np.arctan(df['x0'])
        df['x8'] = -2*np.exp(df['x1']**3)
    
    if classification:
        df['y'] = 0
        df.loc[(df['x0']**2+df['x1']**2+df['x2']**2+df['x3']**2<16) & (df['x0']**2+df['x1']**2+df['x2']**2+df['x3']**2 > 4), 'y'] = 1
        return df
    else:
        df['y'] = -2*np.sin(df['x0']) + np.maximum(df['x1'],0)+df['x2']+np.exp(-df['x3'])
        epsilon = np.random.normal(0,.1, len(df))
        df['y'] += epsilon
        return df
    
def create_model(classification=False, compile_=False):
    if classification:
        # Add the input layer
        inp = keras.layers.Input(shape=(10,))

        # Add several hidden layers
        x = keras.layers.Dense(20, activation='relu')(inp)
        x = keras.layers.Dense(60, activation='relu')(x)
        x = keras.layers.Dense(30, activation='relu')(x)
        x = keras.layers.Dense(20, activation='relu')(x)
        
        # Add the output layer
        out = keras.layers.Dense(2, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=out)
        if compile_:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    else:
        # Add the input layer
        inp = keras.layers.Input(shape=(10,))

        # Add several hidden layers
        x = keras.layers.Dense(20, activation='relu')(inp)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.Dense(50, activation='relu')(x)
        x = keras.layers.Dense(50, activation='relu')(x)
        x = keras.layers.Dense(25, activation='relu')(x)
        
        # Add the output layer
        out = keras.layers.Dense(1, activation='linear')(x)
        model = Model(inputs=inp, outputs=out)
        if compile_:
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

        
        