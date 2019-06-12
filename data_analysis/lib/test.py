# -*- coding: utf-8 -*-
# file       : test.py
# @copyright : MIT
# @purpose   : models testing

# import packages
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import datetime,time
# from sklearn import svm,tree
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier

# import libs && functions
# import lib

__all__ = ("Check","CrossCheck")

# model lists' demo:
# test_models = [{"name":"svm","model":classifier},
#                {"name":"random forest","model":clf},
#                {"name":"decision tree","model":dtree}]



def model_fit_and_predict(model,X_train,y_train,X_test):
    """
    using model to fit data, and return the predicted result
    :param model: scikit-learn model
    :param X_train: pandas.Dataframe traing data of x
    :param y_train: pandas.Dataframe traing data of y
    :param X_test: pandas.Dataframe testing data of x
    :return y_pred: np.ndarry predicted result
    """
    model.fit(X_train,y_train)
    return model.predict(X_test)


def simple_ensemble(models_list,X_train,y_train,X_test):
    """
    using models with sample ensemble techniques to fit data, 
    and return the predicted result
    :param models_list: scikit-learn models list
    :param X_train: pandas.Dataframe traing data of x
    :param y_train: pandas.Dataframe traing data of y
    :param X_test: pandas.Dataframe testing data of x
    :return y_pred: pandas.Series predicted results
    """
    result = {}
    for model_item in models_list:
        model_name = model_item["name"]
        y_pred = model_fit_and_predict(model_item["model"],X_train,y_train,X_test)
        result[model_name] = list(y_pred)
    result = pd.DataFrame(result)
    result["sum"] = result.apply(lambda x: sum(x),axis=1)
    result["vote"]= result.apply(lambda x: 1 if x["sum"] > 0 else -1, axis=1)
    result.index = X_test.index
    return result["vote"].copy()


def model_accuracy(y_test,y_pred):
    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(y_test, y_pred)))
    print("model accuracy: ", metrics.accuracy_score(y_test, y_pred))

def model_test(test_models,epochs,X_train,y_train,X_test,y_test,ensemble=False):
    """
    using models with sample ensemble techniques to fit data, 
    and return the predicted result
    :param test_models: scikit-learn models list
    :param epochs: the loops of testing
    :param X_train: pandas.Dataframe traing data of x
    :param y_train: pandas.Dataframe traing data of y
    :param X_test: pandas.Dataframe testing data of x
    :param ensemble: using ensemble or not, default False
    """

    models_name = []
    for item in test_models:
        models_name.append(item["name"])
    print("Testing these models: ", models_name)
    if ensemble == True:
        print("Using ensemble techniques to predict the stock price movement.")
    print('*'*90)
    if ensemble == False:
        for model_item in test_models:
            print("Using {} model to train and predict the data".format(model_item["name"]))
            for i in range(epochs):
                print("="*80)
                print("ensemble epoch ",i+1)
                y_pred = model_fit_and_predict(model_item["model"],
                                                           X_train,
                                                           y_train,
                                                           X_test)
                model_accuracy(y_test,y_pred)
                print('-'*80)
        return
    if ensemble == True and len(test_models)%2 !=1:
        print("Ensemble needs odd number of models!")
        return
    for i in range(epochs):
        print("ensemble epoch ",i+1)
        print("="*80)
        y_pred = simple_ensemble(test_models,X_train,y_train,X_test)
        model_accuracy(y_test,y_pred)
        print('-'*80)


class CrossCheck(object):
    """using k-fold cross validation to train the model, test the data."""
    def __init__(self, test_models,X,y,cv=5,
                 report_obj=None,ensemble=False,show=True):
        self.test_models = test_models
        self.X = X
        self.y = y
        self.cv = cv
        self.model_check(ensemble)
        self.show = show or False
        self.print_func_list = []
        self.output_functions(report_obj)

    def output_functions(self, report_obj):
        """
        add the report string functions list
        """
        self.print_func_list = []
        if report_obj !=None:
            self.print_func_list.append(report_obj.add)
        if self.show == True:
            self.print_func_list.append(print)        

    def output(self,*output_str):
        """
        description
        """
        output_str = " ".join(str(x) for x in list(output_str))
        if len(output_str) > 0:
            [x(output_str) for x in self.print_func_list]

    def model_check(self,ensemble):
        """
        description
        """
        self.info_str = ""
        if ensemble and len(self.test_models)%2 !=1:
            self.info_str += "Ensemble needs odd number of models!\n"
            self.info_str += "Using k-Fold Cross-Validation techniques to predict the stock price movement.\n"
            self.ensemble = False
        elif ensemble and len(self.test_models)%2 ==1:
            self.info_str +="Using ensemble techniques to predict the stock price movement."
            self.ensemble = True
        else:
            self.info_str += "Using k-Fold Cross-Validation techniques to predict the stock price movement.\n"
            self.ensemble = False

    def seperate_test(self):
        """
        using the models seperately and k-fold cross validation to train the model, test the data.
        """
        for model_item in self.test_models:
            self.output("Using {} model to train and predict the data".format(model_item["name"]))
            self.output("="*80)
            cv_scores = cross_val_score(model_item["model"], self.X, self.y, cv=self.cv)
            self.output("cv_scores list: ",cv_scores)
            self.output("cv_scores mean: {}".format(np.mean(cv_scores)))
            self.output('-'*80)

    def ensemble_test(self):
        """
        using ensemble and k-fold cross validation to train the model, test the data.
        """
        outer_cv = KFold(n_splits=self.cv, shuffle=True, random_state=1)
        cv_scores = []
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y)): 
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            y_pred = simple_ensemble(self.test_models,X_train,y_train,X_test)
            cv_scores.append(accuracy_score(y_test, y_pred))


        self.output("cv_scores list: ",cv_scores)
        self.output("cv_scores mean: {}".format(np.mean(cv_scores)))
        self.output('-'*80)

    def train(self):
        """
        train and testing the data
        """
        models_name = []
        for item in self.test_models:
            models_name.append(item["name"])
        self.output("Testing these models: ", models_name)
        self.output(self.info_str)
        self.output('*'*80)

        if not self.ensemble:
            self.seperate_test()
        else:
            self.ensemble_test()

class Check(object):
    """train the model, test the model, output the results"""
    def __init__(self, test_models,epochs,X,y,test_size=0.2,report_obj=None,ensemble=False,show=True,result_saving_path=None):
        self.test_models = test_models
        self.epochs = epochs

        self.split(X, y, test_size)
        self.show = show or False
        self.print_func_list = []
        self.output_functions(report_obj)
        self.ensemble = ensemble
        self.result_saving_path=result_saving_path

    def output_functions(self, report_obj):
        """
        add the report string functions list
        """
        self.print_func_list = []
        if report_obj !=None:
            self.print_func_list.append(report_obj.add)
        if self.show == True:
            self.print_func_list.append(print)

    def output(self,*output_str):
        """
        description
        """
        output_str = " ".join(str(x) for x in list(output_str))
        if len(output_str) > 0:
            [x(output_str) for x in self.print_func_list]

    def split(self,X,y,test_size=0.2):
        
        """
        Split the dataset into train and test data.
        """
        # get the test data item number
        test_shape = int(X.shape[0]*test_size)
        
        if test_shape != X.shape[0]*test_size:
            test_shape += 1

        train_shape = X.shape[0] - test_shape
        
        self.X_train = X.iloc[0:train_shape-1].copy()
        self.X_test = X.iloc[train_shape:].copy()
        
        self.y_train = y.iloc[0:train_shape-1].copy()
        self.y_test = y.iloc[train_shape:].copy()
        

    def ensemble_test(self):
        """
        using ensemble model to train & test the data
        """
        if len(self.test_models)%2 !=1:
            self.output("Ensemble needs odd number of models!")
            return
        result = {"y_test":self.y_test}
        for i in range(self.epochs):
            self.output("ensemble epoch ",i+1)
            self.output("="*80)
            y_pred = simple_ensemble(self.test_models,
                                                 self.X_train,
                                                 self.y_train,
                                                 self.X_test)
            
            pred_result_key = "epoch_{}".format(i+1)
            result[pred_result_key] = y_pred
            
            self.model_accuracy("ensemble test accuracy: ",self.y_test,y_pred)
            self.output('-'*80)

        if self.result_saving_path:
            self.save_testing_results("ensemble",result)

    def model_accuracy(self,classifier,y_test,y_pred):
        self.output("Classification report for classifier %s:\n%s\n"
            % (classifier, metrics.classification_report(y_test, y_pred)))
        self.output("model accuracy: ", metrics.accuracy_score(y_test, y_pred))

    def save_testing_results(self,model_name,result):
        """
        save the testing result and predicting result.
        """
        now = time.strftime("%Y_%m_%d_%H_%M_%S")
        
        file_path = "{0}/{1}_{2}.csv".format(self.result_saving_path,now,model_name)
        result = pd.DataFrame(result)
        result.to_csv(file_path)

    def seperate_test(self):
        """
        testing the models seperately.
        """
        
        for model_item in self.test_models:
            self.output("Using {} model to train and predict the data".format(model_item["name"]))
            result = {"y_test":self.y_test}
            for i in range(self.epochs):
                self.output("="*80)
                self.output("testing epoch ",i+1)
                y_pred = model_fit_and_predict(model_item["model"],
                                                           self.X_train,
                                                           self.y_train,
                                                           self.X_test)
                pred_result_key = "epoch_{}".format(i+1)
                result[pred_result_key] = y_pred
                self.model_accuracy(model_item["name"],self.y_test,y_pred)
                self.output('-'*80)
            
            if self.result_saving_path:
                self.save_testing_results(model_item["name"],result)

    def train(self):
        """
        train and testing the data
        """
        models_name = []
        for item in self.test_models:
            models_name.append(item["name"])
        self.output("Testing these models: ", models_name)
        if self.ensemble == True:
            self.output("Using ensemble techniques to predict the stock price movement.")
        self.output('*'*90)
        if self.ensemble:
            self.ensemble_test()
        else:
            self.seperate_test()