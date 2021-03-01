import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("/Users/NKAPUR/PycharmProjects/Football_project/data.csv")

data = data.dropna()
for i in range(0,data.shape[0]):
    data['fpl_sel'].iloc[i] = float(str(data['fpl_sel'].iloc[i]).rstrip('%'))
data['fpl_sel'] = data['fpl_sel'].astype('float64')

X = data[data.columns[~data.columns.isin(['market_value'])]]
y = data['market_value']
X = X.drop(['position','name','age','club','nationality'], axis = 1)

print(X.columns)

np.random.seed(37) # Set seed
x_train, x_test = train_test_split(X, test_size = 0.15, random_state = 40 )

y_train = y.loc[x_train.index.values]
y_test = y.loc[x_test.index.values]
x_train = X.loc[x_train.index.values, :]
x_test = X.loc[x_test.index.values, :]

x_train = x_train.dropna()

X = x_train
y = y_train

#### Fitting Random Forest Regression to the dataset
# import the regressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 90, random_state = 0)
# fit the regressor with x and y data
regressor.fit(X, y)

pickle.dump(regressor, open("/Users/NKAPUR/PycharmProjects/Football_project/model.pkl", 'wb'))

