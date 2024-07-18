from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import json
from datetime import datetime, timedelta
import logging
from scrapy.crawler import CrawlerProcess, CrawlerRunner
import os
import re
from textblob import TextBlob
from twisted.internet import reactor, defer
from scrapy.utils.log import configure_logging

import csv


# Base directory of the project (one level above the current file)
BASE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)

# File paths
LAST_UPDATED_FILE = os.path.join(BASE_DIR, r'..\..\Stocks_data\last_updated.txt')


# check if we need update
def needs_update(last_updated_str):
    if not last_updated_str:
        return True
    last_updated = datetime.strptime(last_updated_str, '%Y-%m-%d %H:%M:%S')
    # if the difference is more than a day 
    return datetime.now() - last_updated > timedelta(days=1)

# this runs the web crawler
class YahooCrawler(CrawlSpider):
    name = "yahoocrawler"
    allowed_domains = ["finance.yahoo.com"]

    custom_settings = {
        'LOG_LEVEL': 'INFO',
    }

    def __init__(self, username, *args, **kwargs):
        super(YahooCrawler, self).__init__(*args, **kwargs)
        if username:
            self.start_urls = [f"https://finance.yahoo.com/quote/{username}"]
        else:
            self.start_urls = ["https://finance.yahoo.com/quote/AAPL"]
        self.articles = []
        self.sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    def parse_start_url(self, response):
        article_titles = response.css('h3::text').getall()
        if not article_titles:
            article_titles = response.css('h3 a::text').getall()

        for title in article_titles:
            sentiment = TextBlob(title).sentiment.polarity
            if sentiment > 0:
                self.sentiment_counts["positive"] += 1
            elif sentiment < 0:
                self.sentiment_counts["negative"] += 1
            else:
                self.sentiment_counts["neutral"] += 1

            self.articles.append((title, 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'))
            logging.info(f"Article: {title} | Sentiment: {'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'}")

        logging.info(f"Sentiment summary: {self.sentiment_counts}")

    def closed(self, reason):
        with open(LAST_UPDATED_FILE, 'w') as file:
            file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Run the web crawler and return the results
@defer.inlineCallbacks
def run_spider(username):
    configure_logging()
    runner = CrawlerRunner()
    crawler = runner.create_crawler(YahooCrawler)
    yield runner.crawl(crawler, username=username)
    defer.returnValue((crawler.spider.articles, crawler.spider.sentiment_counts))

def run_crawler(username):
    return run_spider(username).addCallback(lambda results: results)