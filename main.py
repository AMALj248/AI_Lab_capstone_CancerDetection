import sklearn as sks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn import svm
from sklearn import metrics
import seaborn as sns

# importing our cancer dataset
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('data.csv')
print(data)

# doing some basic EDA

# lets get the frequency of cancer categories
sns.countplot(data['diagnosis'], label="Count")
plt.show()

# lets find and remove multi collinearity

corr = data.corr()  # .corr is used for find corelation
plt.figure(figsize=(14, 14))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 15},
            cmap='coolwarm')

plt.show()

# dropping useless coulumns

data.drop("Unnamed: 32", axis=1, inplace=True)
data.columns
data.drop("id", axis=1, inplace=True)

# converting to ordinal values
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# now split our data into train and test
train, test = train_test_split(data, test_size=0.3)  # in this our main data is splitted into train and test
# we can check their dimension
print(train.shape)
print(test.shape)

# selecting our best input variables
prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']

train_X = train[prediction_var]  # taking the training data input
train_y = train.diagnosis  # This is output of our training data
# same we have to do for test
test_X = test[prediction_var]  # taking test data inputs
test_y = test.diagnosis  # output value of test dat

# Base Vanilla Model
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)

print("Prediction Accuracy ", metrics.accuracy_score(prediction, test_y))


# Model with Hyperparameter Tuning

# lets Make a function for Grid Search CV
def Classification_model_gridsearchCV(model, param_grid, data_X, data_y):
    clf = GridSearchCV(model, param_grid, cv=10, scoring="accuracy")

    clf.fit(train_X, train_y)
    print("The best parameter found on development set is :")
    # this will gie us our best parameter to use
    print(clf.best_params_)
    print("the bset estimator is ")
    print(clf.best_estimator_)
    print("The best score is ")
    # this is the best score that we can achieve using these parameters#
    print(clf.best_score_)

# parameter inputs
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10],
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }


# input & output data
data_X= data[prediction_var]
data_y= data["diagnosis"]

# calling the model in function
model= DecisionTreeClassifier()
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)