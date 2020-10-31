# Artificial Neural Network model

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.preprocessing as preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

tf.__version__

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Transforming the gender into numbers
le = preprocessing.LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Transforming the geografical place to numbers
ct = ColumnTransformer(transformers=[('encoder', preprocessing.OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# splitting train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising ANN
ann = tf.keras.models.Sequential()

# Adding hiddens layers
# units are the number of neurons per layer
# "relu" stands for Rectifier activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
# the number of units depends on how many different outputs we may have
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training
# Depending on the output the loss param should be either 'binary_crossentropy' or 'categorical_crossentropy'
# also the activation function should be 'softmax' for more than two expected values
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""
# 0.5 is the threashold we choose to convert the probability to a boolean (>0.5 is true, <0.5 is false)
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

