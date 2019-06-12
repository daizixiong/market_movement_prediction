# -*- coding: utf-8 -*-
# file       : preprocess.py
# @copyright : MIT
# @purpose   : preprocessing the tweets

# import packages
import numpy as np
import pandas as pd
import re
from functools import partial

# import libs && functions
# import lib
# __all__ = (stocks_remove,urls_remove,handles_remove,shorts_remove)
__all__ = ("pattern_remove")

# using reg pattern to remove special characters
def pattern_remove(row,key="text",length=3):
    """
    description
    """
    result = row[key]
    # remove stock mark, like $FB
    result = re.sub(r"\$\w*","",result)

    # remove url string, like "http://amazon.com"
    result = re.sub(r"(http\W+\S+)|(https\W+\S+)","",result)

    # remove twitter handles (@user)
    result = re.sub(r"@[\w]*","",result)

    # Removing Punctuations, Numbers, and Special Characters
    result = re.sub("[^a-zA-Z#]"," ",result)
    # Removing Stop Words
    return shorts_remove(length=length,context=result)

# using pattern to remove special string content function
def pattern_replace(pattern,key,row,replace_text=""):
    target = row[key]
    return re.sub(pattern,replace_text,target)

# remove stock mark, like $FB
stocks_remove = partial(pattern_replace,r"\$\w*",'text',replace_text="")

# remove url string, like "http://amazon.com"
urls_remove = partial(pattern_replace,r"(http\W+\S+)|(https\W+\S+)",'text',replace_text="")

# remove twitter handles (@user)# remove twitter handles (@user)
handles_remove = partial(pattern_replace,r"@[\w]*",'text',replace_text="")

# Removing Short Words# Removing Short Words
def shorts_remove(row=None,key="text",length=3,context=None):
    if context == None:
        target = row[key]
    else:
        target = context

    words_list = target.split()
    words_list = [x for x in words_list if len(x) >= length]
    return ' '.join(words_list)

# Removing Punctuations, Numbers, and Special Characters
# data["text_char_demo"] = data["text"].str.replace("[^a-zA-Z#]", " ")
