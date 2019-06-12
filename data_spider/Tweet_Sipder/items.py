# -*- coding: utf-8 -*-


from scrapy import Item, Field


class STweets(Item):
    id_tweet = Field()
    add_url = Field()
    date_time = Field()
    tweet_content = Field()
    id_user = Field()

    quantity_retweet = Field()
    quantity_favorite = Field()
    quantity_reply = Field()
    quantity_followers = Field()

    is_reply = Field()
    is_retweet = Field()

    company = Field()
