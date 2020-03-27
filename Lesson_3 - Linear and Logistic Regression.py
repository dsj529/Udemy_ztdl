import math

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier

def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
    
## Exercise 1: Housing data
# load data from ../data/housing-data.csv
data = pd.read_csv('../data/housing-data.csv')

# plot a histogram of each feature
for i, col in enumerate(data.columns):
    plt.subplot(1, 4, i+1)
    data[col].plot(kind='hist', title=col, sharey=True)
    plt.xlabel(col)

# split data into X[sqft, bdrms, age], and y[price]
X = data[['sqft', 'bdrms', 'age']].values
y = data['price'].values

# split X and y into 80/20 train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create the linear model
model = Sequential()
model.add(Dense(1, input_shape=(3,)))
model.compile(Adam(lr=0.8), 'mean_squared_error')

model.fit(X_train, y_train, epochs=50)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print('Training set:\nrmse: {:f}, rmsle: {:f}, r2: {:f}'
          .format(math.sqrt(mse(y_train, y_pred_train)),
                  rmsle(y_pred_train, y_train),
                  r2_score(y_train, y_pred_train)))
print('Test set:\nrmse: {:f}, rmsle: {:f}, r2: {:f}'
          .format(math.sqrt(mse(y_test, y_pred_test)),
                  rmsle(y_pred_test, y_test),
                  r2_score(y_test, y_pred_test)))


# create normalized versions of the data columns and try again
data['sqft1000'] = data['sqft'] / 1000.0
data['age10'] = data['age'] / 10.0
data['price100k'] = data['price'] / 1e5

# split data into X[sqft, bdrms, age], and y[price]
X = data[['sqft1000', 'bdrms', 'age10']].values
y = data['price100k'].values

# split X and y into 80/20 train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create the linear model
model = Sequential()
model.add(Dense(1, input_shape=(3,)))
model.compile(Adam(lr=0.8), 'mean_squared_error')

model.fit(X_train, y_train, epochs=50)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print('Training set:\nrmse: {:f}, rmsle: {:f}, r2: {:f}'
          .format(math.sqrt(mse(y_train, y_pred_train)),
                  rmsle(y_pred_train, y_train),
                  r2_score(y_train, y_pred_train)))
print('Test set:\nrmse: {:f}, rmsle: {:f}, r2: {:f}'
          .format(math.sqrt(mse(y_test, y_pred_test)),
                  rmsle(y_pred_test, y_test),
                  r2_score(y_test, y_pred_test)))

    
## Exercise 2: Employee satisfaction
# a) load and describe the data
data = pd.read_csv('../data/HR_comma_sep.csv')
print(data.head())
print(data.info())
print(data.describe())

# b) assume everyone stayed, what is the predictive accuracy?
print(data.left.value_counts() / len(data))
# 76.2%

# c) examine cols for rescaling
data['average_montly_hours'].plot(kind='hist')
data['monthly_100hrs'] = data['average_montly_hours'] / 100.0

# d) convert categorical columns into dummy binary cols
data_dummies = pd.get_dummies(data[['sales', 'salary']])

# e) build X and y matrices 
X = pd.concat([data[['satisfaction_level', 'last_evaluation', 'number_project',
                     'time_spend_company', 'Work_accident', 
                     'promotion_last_5years', 'monthly_100hrs']],
               data_dummies],
              axis=1).values
y = data['left'].values

# f) build the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Sequential()
model.add(Dense(1, input_dim=20, activation='sigmoid'))
model.compile(Adam(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs =50)
y_test_pred = model.predict_classes(X_test)

# g) look at confusion matrix, precision/recall
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# h) test 5x cv
def build_logistic_model():
    model = Sequential()
    model.add(Dense(1, input_dim=20, activation='sigmoid'))
    model.compile(Adam(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_logistic_model, epochs=50, verbose=0)
cv = KFold(5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)

print('The cross-validation accuracy is {:.4f} ± {:.4f}'.format(scores.mean(),
                                                               scores.std()))