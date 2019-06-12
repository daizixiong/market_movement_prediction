# -*- coding: utf-8 -*-
from scrapy.conf import configs
import logging
import pymongo
import json

from OTweets.items import STweets


class SaveToMongoPipeline(object):

    def __init__(self):
        connection = pymongo.MongoClient(configs['MONGODB_SERVER'], configs['MONGODB_PORT'])
        db = connection[configs['MONGODB_DB']]
        self.OYCollect = db[configs['MONGODB_TWEET_COLLECTION']]
        self.OYCollect.ensure_index([('0YID', pymongo.ASCENDING)], unique=True, dropDups=True)
        self.userCollection.ensure_index([('0YID', pymongo.ASCENDING)], unique=True, dropDups=True)


    def process_item(self, item, spider):
        if isinstance(item, STweets):
            OYItem = self.OYCollect.find_one({'0YID': item['0YID']})
            if not OYItem:
                self.OYCollect.insert_one(dict(item))
        elif isinstance(item, User):
            OYItem = self.userCollection.find_one({'0YID': item['0YID']})
            if not OYItem:
                self.userCollection.insert_one(dict(item))