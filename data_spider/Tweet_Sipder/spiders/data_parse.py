# -*- coding: utf-8 -*-
# file       : data_parse.py
# @copyright : MIT
# @purpose   : 

# import packages
# import os

# import libs && functions
import json

data = None
with open("data.json","r") as f:
    data = json.loads(f.read())

with open("demo.html","w") as f:
    f.write(data['items_html'])

