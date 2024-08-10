https://docs.google.com/document/d/1VbzfQNvlcN1ngZXVkpdg4OF71vJQMlBCtzMA5l51sHY/edit






Stock Data Pipeline




Goal
	The goal of this API is to fetch the requested stock data, store the raw data in an Amazon 
S3 bucket, fetch some recent articles about the stock to see its performance and potential 
future performance, clean the data using visualization and principal component analysis, store the condensed data in Aurora as a MySQL serverless table, and return that condensed data to the user.

Extraction

The first step towards this API was the extraction of data. There are 3 main components:
Web Crawler - this is used to scour apple finance for both the url with the company ticker
	For stock data and also to find recent articles about the company to see whether
	Stocks will be positive or negative in the future

S3 buckets - this is used for raw data storage - the csv files from Apple Finance containing both the sentiment analysis of articles and the stock data is stored here

	Web Crawler
		The web crawler is used primarily for retrieval of the article headers for the 
requested stock. This is something I’m using for the future of this api. By running 
sentiment analysis on these headlines, I want to see whether this has a strong correlation to the stock’s future.  


	S3 Buckets 
		The next step was storing this raw data. I chose to use S3 buckets for this storage, 
		Where one bucket is split into two directories. One stores the company’s stock 
data, and the other stores the company’s sentiment data. Since the sentiment data 
is not part of the original dataset, I am not using these for data transformation, as I want to gather more data on them first before using it in the future.

Here is the configuration of my S3 Bucket:

		Rawstockdata 
			Stocks_data
				AAPL_stock_data.csv
				META_stock_data.csv
			Sentiment_data
				AAPL_sentiment_data.csv
				META_sentiment_data.csv

Transformation
	
	The transformation of data first requires some data analysis. Here is the original dataset 
of AAPL_stock_data.csv:


First, I ran tests tracking the Volume of stock vs Date to see if there were any annual cycles:



I did the same with time vs price_change of stock:
	

I also tracked these monthly - to see if there were times of increase or decrease:


Truth be told, there were no obvious correlations besides a slight correlation of stock increase every January, but even that was generous.

The columns themselves didn’t seem to be responsible for much variance in the dataset or yield any patterns, so I decided to run PCA on the dataset to see if transforming the columns could separate the dataset.


After transforming the original dataset - there were the cumulative variances tracked by the new components of the dataset:  [0.76177593, 0.90529276, 0.99977391, 0.99989743, 0.9999615  0.99998692, 1.        ]

Here, the first two components already had 90% of the variance captured, and yielded a strong correlation in the points. Here the points are clearly split. However, I noticed the points are all clustered together, so while there is a distinct separation, I wanted to see if I could increase the variance among these points.

I added more columns: 

Price-Change: calculates the change in price from the start to the end of that day
HighsVLows - tracks the highest point of the stock vs the lowest point of that stock on that day
10_week_averages - tracks the rolling average every 200 days to reduce noise in daily 
Fluctuations
Avg_gain - sees the gain in  stock thru percentages
Month - tracks month
DayOfWeek - tracks day of week 
Month_sin, Month_cos, Day_sin, Day_cos - these were ways to track cycles of time to see if the 
data has some cyclical impact (didn’t yield much)

Then I ran PCA again to see if the points could separate:




Now, there is a clear divide in the points while also having more variance and spread through the data points. I think this suggests that the additional data provides more insights to the dataset which makes distinguishing points easier, so I decided to keep this dataset for cleaning. Here are the cumulative variance values for this dataset:
[0.60886461, 0.78691107, 0.90503113, 0.96180585, 0.98785339, 0.99587285
 0.99998234 0.99999817, 1.   ]

It required more components to reach 90%+ variance, so this suggests that added columns added more complexity to the dataset to track its variance.

So, in summary, to transform the data, I first cleaned any missing values or rows, added additional columns to give insight to the data, and ran PCA on this dataset.

This is stored in an Amazon Aurora mySQL table, one table for each stock. These values are updated every day once whenever the API is called. 

