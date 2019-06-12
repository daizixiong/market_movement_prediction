# coding: utf-8
# file       : predict_and_test_all.py
# @copyright : MIT

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# import algorithm models

from lib.test import Check,CrossCheck
# import report object
from utils.reports import Report
from utils.reports import mkdirs
import datetime,time
import os

if __name__ == '__main__':

    # folders to save log files
    reports_folder = "./log/"
    # folder to save y_test
    result_saving_folder = "./result"
    # raw data files
    data_sets = [{"stock_name":"1","data":"./data/ALL/1.csv"}]

    for dataset in data_sets:

        now = time.strftime("%Y_%m_%d_%H_%M_%S")

        # params path
        params_path = dataset["data"]

        # reports saving path
        stock_reports_folder = "{0}/{1}".format(reports_folder,
                                                dataset["stock_name"])
        mkdirs(stock_reports_folder)

        reports_path = "{0}/{1}.log".format(stock_reports_folder,now)

        # testing result saving path
        result_saving_path = "{0}/{1}".format(result_saving_folder,
                                              dataset["stock_name"])
        mkdirs(result_saving_path)

        report_object = Report(reports_path)


        # import params
        params = pd.read_csv(params_path,index_col="date")

        X = params.loc[:].copy()
        X.drop(["stock_price_movement"],inplace=True,axis=1)
        y = params["stock_price_movement"].copy()

        from lib.models import models

        check = Check(models,3,X,y,test_size=0.2,report_obj=report_object,
                      ensemble=False,show=False,
                      result_saving_path=result_saving_path)

        check.train()

        ensemble_models = models.copy()

        ensemble_models.pop(0)


        ensemble_check = Check(ensemble_models,3,X,y,test_size=0.2,
                               report_obj=report_object,ensemble=True,
                               show=False,
                               result_saving_path=result_saving_path)


        ensemble_check.train()

        cross_check = CrossCheck(models,X,y,cv=10,
                                 report_obj=report_object,show=False)


        cross_check.train()

        cross_ensemble_check = CrossCheck(ensemble_models,X,y,cv=10,
                                          report_obj=report_object,
                                          show=False,ensemble=True)

        cross_ensemble_check.train()