import yfinance as yf
import datetime
import os
from datetime import datetime, timedelta
import pandas as pd
import boto3
import csv
from stocksapi.stocksapi.spiders.Crawler import YahooCrawler, run_crawler
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor, defer

os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA4HZZIVJW3PFTXNUC'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the stock ticker and date range
def api_fetch_stock_data(ticker):
    # first, check if we have that stock_data in our s3 bucket.
    filename = "Stocks_data/" + ticker + "_stock_data.csv"
    if check_s3_bucket(filename):
        download_s3_file(filename)
        if needs_update(ticker):
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = yf.download(ticker, start=start_date, end=end_date)
            data.to_csv(filename)
        else:
            print("file up to date")
    else:
        data = yf.download(ticker, start=(datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
        data.to_csv(filename)
    return filename

def check_s3_bucket(name):
    try:
        filename = "Stocks_data/" + name + "_stock_data.csv"
        s3 = boto3.client('s3')
        s3.head_object(Bucket="rawstockdatav1",Key=filename)
        return True
    except:
        return False
    

def download_s3_file(name):
    filename = "Stocks_data/" + name + "_stock_data.csv"
    s3 = boto3.client('s3')
    s3.download_file("rawstockdatav1", filename, filename)


# sees if we need to update the db
def needs_update(name):
    filename = os.path.join(BASE_DIR, "Stocks_data", f"{name}_stock_data.csv")
    try:
        data = pd.read_csv(filename)
        last_date_str = data['Date'].iloc[-1]
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return True
    return True
    #return (datetime.now() - last_date).days > 1


def update_info(needs_update, name):
    if needs_update:
        filename = api_fetch_stock_data(name)
        data_store_s3(filename, filename)
    
def data_store_s3(filepath, filename):
    s3 = boto3.client('s3')
    s3.upload_file(filepath, "rawstockdatav1", filename)

def sentiment_store_s3(filepath, filename):
    s3 = boto3.client('s3')
    s3.upload_file(filepath, "rawstockdatav1", filename)


@defer.inlineCallbacks
def sentiment_calculator(name):
    filepath = os.path.join(BASE_DIR, "Sentiment_data", f"{name}_sentiment_data.csv")

    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Positive', 'Negative', 'Neutral'])

    if needs_update(name):
        articles, sentiment_counts = yield run_crawler(name)
        
        # Write sentiment analysis results to the file
        current_date = datetime.now().strftime('%Y-%m-%d')
        new_row = [current_date, sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']]
    
        with open(filepath, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

        # S3 bucket config
        filename = "Sentiment_data/" + name + "_sentiment_data.csv"
        sentiment_store_s3(filepath, filename)

def run_sentiment_calculator(name):
    d = sentiment_calculator(name)
    d.addBoth(lambda _: reactor.stop())
    reactor.run()

#update_needed = needs_update('AAPL')
#update_info(update_needed, 'AAPL')


