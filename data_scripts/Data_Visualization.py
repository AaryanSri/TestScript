import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = 'AAPL.csv'  # Update this path to the correct location of your file
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Calculate the days since the first date
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# this column sees how the price differs from year to year (extremely positive)
df['Price_Change'] = df['Close'] - df['Open']

#this column sees how the stock fluctuates in that day (negative)
df['HighsVLows'] = df['High'] - df['Low']

#this initially used to be 10 weeks moving average (barely positive)
df['10_week_averages'] = df['Close'].rolling(window=200).mean()

# this calculates average gain (positive)
df['avg_gain'] = (df['Close'] - df['Open']) / df['Open']


df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek

# Encode cyclic features
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
# Plot the relationship between the days since the first date and the volume of stock
# Calculate the average volume for each month

# Plot the average volume for each month
plt.figure(figsize=(15, 8))
for month in df['Month'].unique():
    monthly_data = df[df['Month'] == month]
    if (month == 12):
     plt.plot(monthly_data['Date'], monthly_data['Price_Change'], marker='o', label=f'Month {month}')

plt.xlabel('Date')
plt.ylabel('Price_Change of Stock')
plt.title('Price_Change Over Time for Each Month')
plt.legend(title='Month')
plt.grid(True)
plt.show()


