# -*- coding: utf-8 -*-
# file       : sentiment.py
# @copyright : MIT
# @purpose   : 

# import packages
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from functools import partial

__all__ = ("get_score","sentiment")


# initalize a sentiment analyser
analyser = SentimentIntensityAnalyzer()

# setting the data file path
def get_score(text,key,row):
    score = analyser.polarity_scores(row[text])
    return score[key]


def sentiment(row):
    """
    description
    """
    if row["compound"] > 0:
        return 1
    elif row["compound"] < 0:
        return -1
    else:
        return 0


# compound = partial(get_score,"text","compound")
# negative = partial(get_score,"text","neg")
# neutral = partial(get_score,"text","neu")
# positive = partial(get_score,"text","pos")

if __name__ == '__main__':

    # process the data & get the sentiment scores
    tweet_data["compound"] = tweet_data.apply(compound,axis=1)
    tweet_data["neg"] = tweet_data.apply(negative,axis=1)
    tweet_data["pos"] = tweet_data.apply(positive,axis=1)
    tweet_data["neu"] = tweet_data.apply(neutral,axis=1)

    tweet_file_path = "./data/tweet.csv"

    # import csv format data
    tweet_data = pd.read_csv(tweet_file_path)

    # setting the file path to save the results
    tweet_file_save_path = "./results/tweets_sentiment.csv"

    # save the results
    tweet_data.to_csv(tweet_file_save_path)
