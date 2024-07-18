import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

apple_stocks = pd.read_csv("AAPL.csv")

# use this field to measure accuracy
apple_stocks['Answer'] = (apple_stocks['Close'] > apple_stocks['Open']).astype(int)

#prior to data transformation, the accuracy is 0.6


# first, lets add a way for the model to track time. 
apple_stocks['Date'] = pd.to_datetime(apple_stocks['Date'])
reference_date = pd.to_datetime("2019-07-01")

apple_stocks['Days'] = (apple_stocks['Date'] - reference_date).dt.days
# here are the new columns

# this column sees how the price differs from year to year (extremely positive)
apple_stocks['Price_Change'] = apple_stocks['Close'] - apple_stocks['Open']

#this column sees how the stock fluctuates in that day (negative)
apple_stocks['HighsVLows'] = apple_stocks['High'] - apple_stocks['Low']

#this initially used to be 10 weeks moving average (barely positive)
apple_stocks['10_week_averages'] = apple_stocks['Close'].rolling(window=200).mean()

# this calculates average gain (positive)
apple_stocks['avg_gain'] = (apple_stocks['Close'] - apple_stocks['Open']) / apple_stocks['Open']

#just removes any na values (likely no na values)
apple_stocks.dropna(inplace=True)

#using these features
features = ['Days','Open', 'High', 'Low', 'Volume', 'avg_gain', 'Price_Change']

#set x and y for random forest
X = apple_stocks[features]
Y = apple_stocks['Answer']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

# used random forest because it is kind of better than a decision tree. It's not as likely
# to overfit to the data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("===================================================================")
print(f"Accuracy of Trained Model on Test 1: {accuracy}")
print("Classification Report (Price_Change, avg_gain, and base columns):")
print(report)



#Here are my trials


# Test 1: Seeing which columns help the most
  # One issue is that, no matter how I split the training and testing, I'm now getting 100% accuracy
  # this is cool but its not realistic

  # i tried changing variables to see if it would make any difference. Here are the results.

  # if I only run the model using the Price_Change column to help, accuracy is 100%

  # if I only run the model using only HighsVLows column to help, accuracy is 59%. Even worse, so this column is useless

  # if I only run the model using 10_week_averages, the accuracy went up to 61%. However, this is only by using 200-moving-averages

  # if I only run the model using avg_gain, it gave me 100% accuracy. 

#Test 2: Now let's see how good this model is at predicting the future.

    # I will run the same model again, but now it can only predict using the days column.

X_test_only_days_since_ref = X_test.copy()
X_test_only_days_since_ref.loc[:, X_test_only_days_since_ref.columns != 'Days'] = 0

# Make predictions
y_pred_only_days_since_ref = rf_model.predict(X_test_only_days_since_ref)

# Evaluate the model
accuracy_only_days_since_ref = accuracy_score(y_test, y_pred_only_days_since_ref)
report_only_days_since_ref = classification_report(y_test, y_pred_only_days_since_ref, zero_division=0)

print("===================================================================")
print(f"Accuracy for Test2: {accuracy_only_days_since_ref}")
print("Classification Report using only 'Days':")
print(report_only_days_since_ref)



# now, my accuracy is only 46% (for daily dataset). Since, in the future, we won't have access
# to price data, I'll try to increase my accuracy by finding patterns within the dates itself


# Test 3: adding date-wise columns

#plonting month and day_of_week numbers to see if the model might find some sort of pattern.
apple_stocks['Month'] = apple_stocks['Date'].dt.month
apple_stocks['Day_of_Week'] = apple_stocks['Date'].dt.dayofweek

# Encode cyclic features
apple_stocks['Month_sin'] = np.sin(2 * np.pi * apple_stocks['Month'] / 12)
apple_stocks['Month_cos'] = np.cos(2 * np.pi * apple_stocks['Month'] / 12)
apple_stocks['Day_sin'] = np.sin(2 * np.pi * apple_stocks['Day_of_Week'] / 7)
apple_stocks['Day_cos'] = np.cos(2 * np.pi * apple_stocks['Day_of_Week'] / 7)

# now let's restart. Let's retrain the model with these features too

features = ['Days','Open', 'High', 'Low', 'Volume', 'avg_gain', 'Price_Change', 'Month', 'Day_of_Week', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']

#set x and y for random forest
X = apple_stocks[features]
Y = apple_stocks['Answer']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

# used random forest because it is kind of better than a decision tree. It's not as likely
# to overfit to the data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("===================================================================")
print(f"Accuracy when training on everything + time-based data: {accuracy}")
print("Classification Report with time based data and everything:")
print(report)


# let's now see how good it is at predicting the future
# Evaluate the model using only specific date-related columns
# Create a new DataFrame with the same feature columns but with only selected date-related columns filled in
X_test_selected_features = X_test.copy()
for column in X_test_selected_features.columns:
    if column not in ['Days', 'Month_sin', 'Day_of_Week', 'Month_cos', 'Day_sin', 'Day_cos']:
        X_test_selected_features[column] = 0

# Make predictions
y_pred_selected_features = rf_model.predict(X_test_selected_features)

# Evaluate the model
# Evaluate the model
accuracy_selected_features = accuracy_score(y_test, y_pred_selected_features)
report_selected_features = classification_report(y_test, y_pred_selected_features, zero_division=0)

print("===================================================================")
print(f"Accuracy using only date-related columns: {accuracy_selected_features}")
print("Classification Report using only date-related columns:")
print(report_selected_features)

# so unfortunately, the dates don't have a good correlation with stock price increases or decreases.
# there doens't seem to be any cyclic patterns with stock in terms of months or days
# 
# let's try another approach


# Test 4: The stock market

# let's see If the model does a good job if it's given context on how the stock market is doing

sp500 = pd.read_csv("S&P500.csv")

# Convert S&P 500 date to datetime
sp500['Date'] = pd.to_datetime(sp500['Date'])

# Ensure the S&P 500 data has the correct column for increases (assume it's named 'SP500_Increase')
# For this example, let's assume you have already created this column in your S&P 500 DataFrame
sp500['SP500_Increase'] = (sp500['Close/Last'] > sp500['Open']).astype(int)

apple_stocks = pd.merge(apple_stocks, sp500[['Date', 'SP500_Increase']], on='Date', how='left')

#using these features
features = ['Days','Open', 'High', 'Low', 'Volume', 'avg_gain', 'Price_Change', 'SP500_Increase']

#set x and y for random forest
X = apple_stocks[features]
Y = apple_stocks['Answer']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

# used random forest because it is kind of better than a decision tree. It's not as likely
# to overfit to the data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("===================================================================")
print("===================================================================")
print("===================================================================")
print("===================================================================")
print("===================================================================")
print(f"Accuracy with everything + SPF 500: {accuracy}")
print("Classification Report:")
print(report)

X_test_selected_features = X_test.copy()
for column in X_test_selected_features.columns:
    if column not in ['Days', 'SP500_Increase']:
        X_test_selected_features[column] = 0

# Make predictions
y_pred_selected_features = rf_model.predict(X_test_selected_features)

# Evaluate the model
# Evaluate the model
accuracy_selected_features = accuracy_score(y_test, y_pred_selected_features)
report_selected_features = classification_report(y_test, y_pred_selected_features, zero_division=0)

print("===================================================================")
print(f"Accuracy using only sp500: {accuracy_selected_features}")
print("Classification Report using only sp500:")
print(report_selected_features)

print(apple_stocks.head())