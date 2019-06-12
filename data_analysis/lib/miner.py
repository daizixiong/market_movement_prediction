# -*- coding: utf-8 -*-
# file       : miner.py
# @copyright : MIT
# @purpose   : mine the factor from cleaned and processed data

# import packages
import numpy as np
import pandas as pd

# import libs && functions
# import lib

__all__ =("sentiment_distribution")

def sentiment_distribution(raw_data):
    """
    get the sentiment distribution from the raw data
    :param raw_data: pandas.Dataframe cleaned & processed data
    :return: pandas.Dataframe sentiment_distribution 
    example of the output sentiment_distribution:

        datetime_format no_reply_neg no_reply_neu no_reply_pos replied_neg replied_neu replied_pos
    0   2018-01-01 0.1070 0.0642 0.2193 0.1818 0.0535 0.3743
    """
    data = raw_data.copy()
    pivot = pd.pivot_table(data,
        index=["datetime_format"],
        columns=["is_reply","sentiment"],
        values=["ID"],aggfunc=[lambda x: len(x)])
    pivot["total"] = pivot.apply(np.sum,axis=1)
    pivot.columns = pivot.columns.droplevel([0,1])
    pivot.columns = pivot.columns.map(lambda x :"_".join([str(m) for m in x]))
    pivot.columns=["no_reply_neg","no_reply_neu",
                                 "no_reply_pos","replied_neg",
                                 "replied_neu","replied_pos","total"]

    sentiment_total = pivot.loc[:,"total"].copy()
    normal_data = pivot.div(sentiment_total,axis=0)
    normal_data = normal_data.applymap(lambda x: '%.4f' % x)

    return normal_data.drop(["total"],axis=1).reset_index().copy()

def market_sentiment(raw_data):
    """
    mine the market sentiment from the raw data
    :param raw_data: pandas.Dataframe cleaned & processed data
    :return: pandas.Dataframe market_sentiment

    example of the output sentiment_distribution: 

    """
    # TODO
    pass