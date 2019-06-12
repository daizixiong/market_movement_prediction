USER_AGENT = 'wangzhe1289@gmail.com'

BOT_NAME = 'OTweets'
LOG_LEVEL = 'WARN'
DOWNLOAD_HANDLERS = {'s3': None,}

SPIDER_MODULES = ['OTweets.spiders']
NEWSPIDER_MODULE = 'OTweets.spiders'
ITEM_PIPELINES = {
    'OTweets.pipelines.SaveToMongoPipeline':100
}
MONGODB_SERVER = "127.0.0.1"
MONGODB_PORT = 27017
MONGODB_DB = "tweets"      
MONGODB_TWEET_COLLECTION = "tweet"
MONGODB_USER_COLLECTION = "user" 


