import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
apple_stocks = pd.read_csv("AAPL2.csv")

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


mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions
y_pred_mlp = mlp_model.predict(X_test)

# Evaluate the model
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
report_mlp = classification_report(y_test, y_pred_mlp, zero_division=0)

print("===================================================================")
print(f"Neural Network Accuracy: {accuracy_mlp}")
print("Neural Network Classification Report:")
print(report_mlp)
X_test_only_days_since_ref = X_test.copy()
X_test_only_days_since_ref.loc[:, X_test_only_days_since_ref.columns != 'Days'] = 0

# Make predictions
y_pred_only_days_since_ref = mlp_model.predict(X_test_only_days_since_ref)

# Evaluate the model
accuracy_only_days_since_ref = accuracy_score(y_test, y_pred_only_days_since_ref)
report_only_days_since_ref = classification_report(y_test, y_pred_only_days_since_ref, zero_division=0)

print("===================================================================")
print(f"Accuracy for Test2: {accuracy_only_days_since_ref}")
print("Classification Report using only 'Days':")
print(report_only_days_since_ref)

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
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = mlp_model.predict(X_test)

# Evaluate the model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb, zero_division=0)

print("===================================================================")
print(f"Gradient Boosting Accuracy Part 3 with Time-based cols: {accuracy_gb}")
print("Gradient Boosting Classification Report:")
print(report_gb)


X_test_selected_features = X_test.copy()
for column in X_test_selected_features.columns:
    if column not in ['Days', 'Month_sin', 'Day_of_Week', 'Month_cos', 'Day_sin', 'Day_cos']:
        X_test_selected_features[column] = 0

# Make predictions
y_pred_selected_features = mlp_model.predict(X_test_selected_features)

# Evaluate the model
# Evaluate the model
accuracy_selected_features = accuracy_score(y_test, y_pred_selected_features)
report_selected_features = classification_report(y_test, y_pred_selected_features, zero_division=0)

print("===================================================================")
print(f"Accuracy using only date-related columns: {accuracy_selected_features}")
print("Classification Report using only date-related columns:")
print(report_selected_features)
