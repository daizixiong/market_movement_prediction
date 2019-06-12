# -*- coding: utf-8 -*-
# file       : processing.py
# @Author    : Zhe 
# @copyright : MIT
# @purpose   : tweets preprocessing

# import packages
import numpy as np
import pandas as pd
from functools import partial

# import from local libs
from lib.preprocess import pattern_remove
from lib.sentiment import get_score



if __name__ == '__main__':
    test_data_filepath = "./data/tweet.csv"
    data = pd.read_csv(test_data_filepath)

    # clean the text content
    data["text_clean"] = data.apply(pattern_remove,axis=1)
    
    compound = partial(get_score, "text_clean", "compound")
    negative = partial(get_score, "text_clean", "neg")
    neutral = partial(get_score, "text_clean", "neu")
    positive = partial(get_score, "text_clean", "pos")

    # using vader to analysis the context sentiment    
    data["compound"] = data.apply(compound, axis=1)
    data["negative"] = data.apply(negative, axis=1)
    data["neutral"] = data.apply(neutral, axis=1)
    data["positive"] = data.apply(positive, axis=1)

    # add stocks and company infos to the data record
    data["stock"] = "AMZN"
    data["company"] = "Amazon"

    # format the date record
    data["datetime_format"] = pd.to_datetime(data["datetime"]).dt.date

    # save the processed data
    processed_data_save_path = "./result/preprocessed_data.csv"
    data.to_csv(processed_data_save_path)