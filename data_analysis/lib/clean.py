# -*- coding: utf-8 -*-
# file       : clean.py
# @copyright : MIT
# @purpose   : cleaner

# import packages
# import os

# import libs && functions
from preprocess import pattern_remove
from sentiment import get_score
from functools import partial

__all__ = ("Clean")

class Clean(object):
    """cleaning object for the data processor"""
    def __init__(self, key="text_clean"):
        self.key = key
        self.compound = partial(get_score, key, "compound")
        self.negative = partial(get_score, key, "neg")
        self.neutral = partial(get_score, key, "neu")
        self.positive = partial(get_score, key, "pos")

    del clean(self,raw_data):
        """
        clean the input raw data
        :param raw_dta: pandas.Dataframe tweets raw data
        """
        data = raw_data.copy()
        data[self.key] = data.apply(pattern_remove,axis=1)

        # using vader to analysis the context sentiment    
        data["compound"] = data.apply(compound, axis=1)
        data["negative"] = data.apply(negative, axis=1)
        data["neutral"] = data.apply(neutral, axis=1)
        data["positive"] = data.apply(positive, axis=1)

        # add sentiment class
        data["sentiment"] = data.apply(sentiment, axis=1)

        # add favoriate or not
        data["is_favorite"] = data.apply(is_favorite, axis=1)

        # change te is_reply record
        data["is_reply"] = data["is_reply"].astype(int)

        return data.copy()

    @staticmethod
    def sentiment(row):
        compound = row["compound"]
        if compound > 0:
            return 1
        elif compound < 0:
            return -1
        return 0

    @staticmethod
    def is_favorite(row):
        favorite=int(row["nbr_favorite"])
        if favorite > 0:
            return 1
        else:
            return -1