# -*- coding: utf-8 -*-
# file       : sentiment_analysis.py
# @copyright : MIT
# @purpose   : 

# import packages
import numpy as np
import pandas as pd
from functools import partial

from lib.preprocess import pattern_remove
from lib.sentiment import get_score
from lib.sentiment import sentiment

if __name__ == '__main__':

    source_folder_path = "~/Desktop/a_new_tweet/"
    save_foler_path = "~/Desktop/1_sentiment_result/"

    files = [{"source":"fb"}]

    # preprocese or not
    preprocese_on = True

    for item in files:
        tweet_file_path = source_folder_path + item["source"] + '.csv'
        tweet_file_save_path = save_foler_path + item["source"] + '_sentiment.csv'


        # import csv format data
        tweet_data = pd.read_csv(tweet_file_path)

        if preprocese_on:
            tweet_data["text_clean"] = tweet_data.apply(pattern_remove,axis=1)
            key = "text_clean"
        else:
            key = "text"

        compound = partial(get_score,key,"compound")
        negative = partial(get_score,key,"neg")
        neutral = partial(get_score,key,"neu")
        positive = partial(get_score,key,"pos")

        # process the data & get the sentiment scores
        tweet_data["compound"] = tweet_data.apply(compound,axis=1)
        tweet_data["neg"] = tweet_data.apply(negative,axis=1)
        tweet_data["pos"] = tweet_data.apply(positive,axis=1)
        tweet_data["neu"] = tweet_data.apply(neutral,axis=1)
        tweet_data["sentiment"] = tweet_data.apply(sentiment,axis=1)

        # setting the file path to save the results

        # save the results
        tweet_data.to_csv(tweet_file_save_path)
