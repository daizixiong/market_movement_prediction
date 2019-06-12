# -*- coding: utf-8 -*-
# file       : market_sentiment.py
# @copyright : MIT

# import packages
import numpy as np
import pandas as pd
from functools import partial

# import libs && functions
# import lib

if __name__ == '__main__':

    source_folder_path = "./data/tweet_results/"
    save_foler_path = "./data/tweet_market_sentiment/"

    files = [{"source":"aapl_sentiment"},
             {"source":"amzn_sentiment"},
             {"source":"csco_sentiment"},
             {"source":"fb_sentiment"},
             {"source":"gild_sentiment"},
             {"source":"goog_sentiment"},
             {"source":"googl_sentiment"},
             {"source":"msft_sentiment"},
             {"source":"nflx_sentiment"},
             {"source":"nvda_sentiment"},
             {"source":"sbux_sentiment"},
             {"source":"tsla_sentiment"}]
    		

    for item in files:
        # set source file path
        tweet_file_path = source_folder_path + item["source"] + '.csv'

        # set the file path to save the results
        tweet_file_save_path = save_foler_path + item["source"] + '_market.csv'

        # import csv format data
        tweet_data = pd.read_csv(tweet_file_path)
        # process the data & get the sentiment scores
        # tweet_data["compound"] = tweet_data.apply(compound,axis=1)
        # tweet_data["neg"] = tweet_data.apply(negative,axis=1)
        # tweet_data["pos"] = tweet_data.apply(positive,axis=1)
        # tweet_data["neu"] = tweet_data.apply(neutral,axis=1)

        tweet_data["marke_sentiment"] = tweet_data["Followers"] * tweet_data["compound"]

        # save the results
        tweet_data.to_csv(tweet_file_save_path)
