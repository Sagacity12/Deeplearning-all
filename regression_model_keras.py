import  os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

import numpy as np
import pandas as pd
import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import Input



import  warnings
warnings.simplefilter('ignore', FutureWarning)

filepath = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)


# checking how many datapoint
# concrete_data.shape
concrete_data.describe()
concrete_data.isnull().sum()
print(concrete_data)

# split data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']
print(predictors.head())
print(target.head())

predictors_norm = (predictors - predictors.mean()) / predictors.std()
print(predictors_norm.head())

n_cols = predictors_norm.shape[1]

# Building a Neural Network
# define regression model
def regression_model():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
         # compile model
    model.compile(optimizer='adam', loss='mse')
    return model

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.2, epochs=100, verbose=1)
