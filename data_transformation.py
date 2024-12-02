import yfinance as yf
import os
from datetime import datetime, timedelta
import pandas as pd
import boto3
import csv
from stocksapi.stocksapi.spiders.Crawler import run_crawler
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor, defer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine


TABLE_NAME = "(TICKER)_DATA"
TABLE_FORMAT = "Date    PC1   PC2   PC3   TARGET"

def clean_data(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.ffill()
    df['target'] = (df['Close'] > df['Open']).astype(int)
    df.drop_duplicates(inplace=True)
    return df

def add_data(df):
    # these are analytical additions to help with data analysis
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Price_Change'] = df['Close'] - df['Open']
    df['HighsVLows'] = df['High'] - df['Low']
    df['10_week_averages'] = df['Close'].rolling(window=200).mean()
    df['avg_gain'] = (df['Close'] - df['Open']) / df['Open']
    # add time + cyclical cycles for potential patterns
    df['Month'] = df['Date'].dt.month
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
    return df

def pca_analysis(df, variance_threshold=0.95):
    numerical_data = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Days', 
                        'Price_Change', 'HighsVLows', '10_week_averages', 'avg_gain']]

    # Impute missing values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(numerical_data)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_imputed)

    # Apply PCA
    pca = PCA()
    pca.fit(scaled_data)

    # Calculate explained variance ratio and cumulative variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()
    print(explained_variance_ratio)
    print(cumulative_variance)
    # Determine the number of components needed to reach the variance threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Transform the data to principal components with the required number of components
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    # Create a DataFrame with the selected principal components
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
    pca_df['target'] = df['target']
    pca_df['Date'] = df['Date']
    selected_columns = ['Date', 'PC1', 'PC2', 'PC3', 'target']
    pca_df = pca_df[selected_columns]
    pca_df.to_csv("test.csv")

    #print("Loadings:")
    #print(loadings_df)
    #print("\nExplained Variance Ratio:", explained_variance_ratio)
    #print("Cumulative Variance:", cumulative_variance)
    #print(f"\nNumber of components selected to reach {variance_threshold*100}% variance: {n_components}")
    #print(pca_df)
    return pca_df

def connect_to_writer_db():
    engine = create_engine(
        'aurora engine'
    )
    return engine

def write_to_db(df, table_name, engine):
 
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)


def connect_to_reader_db():
    engine = create_engine(
        'aurora engine'
    )
    return engine

def read_from_db(table_name, engine):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con=engine)
    return df
