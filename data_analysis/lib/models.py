# -*- coding: utf-8 -*-
# file       : models.py
# @copyright : MIT
# @purpose   : scikit-learn models and functions

# import packages
from sklearn import svm,tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# import libs && functions
# import lib

__all__ = ("models")

# model lists' demo:
# test_models = [{"name":"svm","model":classifier},
#                {"name":"random forest","model":clf},
#                {"name":"decision tree","model":dtree}]

# svm model
classifier = svm.SVC(gamma=0.001)

# naive bayes
gnb = GaussianNB()

# random forest
clf = RandomForestClassifier(n_estimators=40)

# decision tree
dtree = tree.DecisionTreeClassifier()

# logistic regression
lg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

# k Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)

models = [{"name":"svm","model":classifier},
          {"name":"navie_bayes","model":gnb},
          {"name":"random_forest","model":clf},
          {"name":"logistic_regression","model":lg},
          {"name":"k_nearest_neighbors","model":knn},
          {"name":"decision_tree","model":dtree},]