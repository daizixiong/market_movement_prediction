# coding: utf-8

# using the params given the train the model & testing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# import algorithm models
from lib.models import models

# import report object
from utils.reports import Report


# settings
# 1. params file path
# 2. report saving path
# 
# 
# ----



# params path
params_path = "./result/data_set.csv"

# reports saving path
reports_path = "./log/test1888.log"




report_object = Report(reports_path)




# import params
params = pd.read_csv(params_path)




params.head()


## create training & testing data



X = params.loc[:].copy()
X.drop(["stock_price_movement","datetime_format"],inplace=True,axis=1)
y = params["stock_price_movement"].copy()


## 分割数据集
# 
# ---



# 




from lib.test import Check,CrossCheck

check = Check(models,3,X,y,test_size=0.2,random_state=8,report_obj=report_object,ensemble=False)

# check.train()


cross_check = CrossCheck(models,X,y,cv=5,report_obj=report_object)

cross_check.train()


















