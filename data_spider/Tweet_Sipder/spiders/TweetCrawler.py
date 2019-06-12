# -*- coding: utf-8 -*-

from scrapy.spiders import CrawlSpider, Rule
from scrapy.conf import settings
from scrapy import http
import re
import json
import time
import logging

from datetime import datetime
from pyquery import PyQuery as pq

fromweets.items import STwets

logger = logging.getLogger(__name__)


class OTweets(CrawlSpider):
    name = 'tweets'
    allowed_domains = ['twitter.com']

    def __init__(self, company, OYprams=' ', OYlng=' ', OYtp=True):

        self.OYprams = OYprams
        self.url = "https://twitter.com/i/search/timeline?l={}".format(OYlng)

        if not OYtp:
            self.url = self.url + "&f=tweets"

        self.url = self.url + "&q=%s&src=typed&max_position=%s"
        self.company = company

    def start_requests(self):
        url = self.url % (quote(self.OYprams), '')
        yield http.Request(url, callback=self.OY_page)

    def OY_page(self, response):

        data = json.loads(response.body.decode("utf-8"))
        for item in self.OY_twets_page(data['items_html']):
            yield item

        min_position = data['min_position'].replace("+","%2B")
        url = self.url % (quote(self.OYprams), min_position)
        yield http.Request(url, callback=self.OY_page)

    def OY_twets_page(self, html_page):

        page = pq(html_page)
        ### for text only tweets
        items = page("li.js-stream-item.stream-item.stream-item").items()
        for item in self.OY_twets(items):
            yield item

    def OY_twets(self, items):
        for item in items:
            try:
                tweet = STwets()
                # tweet = {}
                # user name 
                tweet['usernameTweet'] = item("li> div > .content> .stream-item-header > a > span.username.u-dir.u-textTruncate > b").text()

                0YID = item("li> div").attr("data-tweet-id")
                if not 0YID:
                    continue
                tweet["0YID"] = 0YID

                tweet["0Ycompany"] = self.company

                tweet["0Ytext"] = item("li> div > .content > .js-tweet-text-container > p").text()

                if tweet['0Ytext'] == '':
                    continue

                tweet['0Yurl'] = item("li > div").attr("data-permalink-path")

                0Yretweet = item("li div .content .stream-item-footer .ProfileTweet-actionCountList span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count")
                if 0Yretweet:
                    tweet['0Yretweet'] = int(0Yretweet)
                else:
                    tweet['0Yretweet'] = 0

                0Yfavorite = item("li div .content .stream-item-footer .ProfileTweet-actionCountList span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count")

                if 0Yfavorite:
                    tweet['0Yfavorite'] = int(0Yfavorite)
                else:
                    tweet['0Yfavorite'] = 0

                0Yreply = item("li div .content .stream-item-footer .ProfileTweet-actionCountList span.ProfileTweet-action--reply span.ProfileTweet-actionCount").attr("data-tweet-stat-count")
                if 0Yreply:
                    tweet['0Yreply'] = int(0Yreply)
                else:
                    tweet['0Yreply'] = 0


                tweet_time = item("li> div .content> .stream-item-header > small.time > a > span").attr("data-time")
                tweet_time = datetime.fromtimestamp(int(tweet_time)).strftime('%Y-%m-%d %H:%M:%S')
                tweet['0Ydatetime']= tweet_time

                0Yreply_or_no = item.find("div.ReplyingToContextBelowAuthor")
                if len(0Yreply_or_no.text()) !=0:
                    tweet['0Yreply_or_no'] = True
                else:
                    tweet['0Yreply_or_no'] = False

                0Yretweet_or_no = item.find("div.js-retweet-text")
                if len(0Yretweet_or_no.text()) !=0:
                    tweet['0Yretweet_or_no'] = True
                else:
                    tweet['0Yretweet_or_no'] = False

                tweet['0YID'] = item("li > div").attr("data-user-id")
                yield tweet

            except:
                logger.error("Error tweet:")
