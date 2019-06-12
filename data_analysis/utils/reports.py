# -*- coding: utf-8 -*-
# file       : reports.py
# @copyright : MIT
# @purpose   : utils for testing and report saving class

# import packages
import os

# import libs && functions
# import lib

class Report(object):
    """for saving the testing report"""
    def __init__(self, file):
        self.file = file

    def add(self,record):
        """
        add testing record to the target file
        """
        record_str = str(record)
        if record_str == None or len(record_str) == 0:
            return
        else:
            with open(self.file,'a') as f:
                f.write(record_str +"\n")

    def header(self,params):
        """
        create the testing header
        """
        pass


def mkdirs(folder):
    """
    create folder
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

