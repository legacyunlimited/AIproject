import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')  # For not showing model warnings to the user

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from menu import get_details


# Loading the dataset
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

'''Data Preprocessing'''

# Here we are fixing the missing values in the dataset.
df_train['Gender'] = df_train['Gender'].fillna(
    df_train['Gender'].dropna().mode().values[0])
df_train['Married'] = df_train['Married'].fillna(
    df_train['Married'].dropna().mode().values[0])
df_train['Dependents'] = df_train['Dependents'].fillna(
    df_train['Dependents'].dropna().mode().values[0])
df_train['Self_Employed'] = df_train['Self_Employed'].fillna(
    df_train['Self_Employed'].dropna().mode().values[0])
df_train['LoanAmount'] = df_train['LoanAmount'].fillna(
    df_train['LoanAmount'].dropna().median())
df_train['Loan_Amount_Term'] = df_train['Loan_Amount_Term'].fillna(
    df_train['Loan_Amount_Term'].dropna().mode().values[0])
df_train['Credit_History'] = df_train['Credit_History'].fillna(
    df_train['Credit_History'].dropna().mode().values[0])


# Here we are converting the values to numeric to get ready for modelling.
code_numeric = {'Male': 1, 'Female': 2,
                'Yes': 1, 'No': 2,
                'Graduate': 1, 'Not Graduate': 2,
                'Urban': 3, 'Semiurban': 2, 'Rural': 1,
                'Y': 1, 'N': 0,
                '3+': 3}

df_train = df_train.applymap(
    lambda s: code_numeric.get(s) if s in code_numeric else s)
df_test = df_test.applymap(
    lambda s: code_numeric.get(s) if s in code_numeric else s)

# dropping the unique loan id
df_train.drop('Loan_ID', axis=1, inplace=True)

# We  need to convert 'Dependents' feature to numeric using pd.to_numeric
Dependents_ = pd.to_numeric(df_train.Dependents)
Dependents__ = pd.to_numeric(df_test.Dependents)
df_train.drop(['Dependents'], axis = 1, inplace = True)
df_test.drop(['Dependents'], axis = 1, inplace = True)
df_train = pd.concat([df_train, Dependents_], axis = 1)
df_test = pd.concat([df_test, Dependents__], axis = 1)

# Separaing target and features in the dataset.
y = df_train['Loan_Status']
X = df_train.drop('Loan_Status', axis = 1)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Building a Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)   # Fitting the model with train dataset
ypred = lr_model.predict(X_test)   # Predicting the values with test dataset
evaluation = f1_score(y_test, ypred)


# Building a Decision Tree model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train) # Fitting the model with train dataset

ypred_tree = tree.predict(X_test) # Predicting the values with test dataset
evaluation_tree = f1_score(y_test, ypred_tree)

''' Tensorflow Keras Library '''

# Initializing ANN
ann = tf.keras.models.Sequential()
# adding input and first hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
# adding second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# adding third hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# adding output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# compiling ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# training ANN on training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 150, verbose=0)
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
eval = accuracy_score(y_test, y_pred)


# To clear out tensorflow's comments.
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

dtt = get_details()
xtt = pd.DataFrame.from_dict(dtt)
ypred_lr = lr_model.predict(xtt)
ypred_tree = tree.predict(xtt)
y_pred_ann = ann.predict(xtt)
y_pred_ann = (y_pred_ann > 0.5)
Check = {
    1:"Yes",
    0:"No"
}
print("\nPrediction with 3 different algorithms:\n")
print("1. Logistic regression (Sklearn) (Accuracy: {}) : {} ".format(round(evaluation,2),Check[ypred_lr[0]]))
print("2. Decision Tree (Sklearn) (Accuracy: {}) : {} ".format(round(evaluation_tree,2),Check[ypred_tree[0]]))
print("3. Artificial Neural Network (Tensorflow) (Accuracy: {}) : {} ".format(round(eval,2),Check[y_pred_ann[0][0]]))


