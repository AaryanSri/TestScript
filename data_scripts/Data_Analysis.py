import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer


df = pd.read_csv('Stocks_data/AAPL_stock_data.csv')
df['target'] = (df['Close'] > df['Open']).astype(int)


numerical_data = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'target']]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

pca = PCA()
pca.fit(scaled_data)

pca_data = pca.transform(scaled_data)


pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

print("Explained Variance Ratio:", explained_variance_ratio)
print("Cumulative Variance:", cumulative_variance)

plt.figure(figsize=(10, 7))
plt.scatter(pca_data[:,0], pca_data[:,1], c=df['target'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of AAPL Data')
plt.grid(True)
plt.show()

loadings = pca.components_

loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=numerical_data.columns)

print(loadings_df)

# add more dfs columns

print("====================================================")
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
df['Price_Change'] = df['Close'] - df['Open']
df['HighsVLows'] = df['High'] - df['Low']
df['10_week_averages'] = df['Close'].rolling(window=200).mean()
df['avg_gain'] = (df['Close'] - df['Open']) / df['Open']
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)


# Select numerical columns for PCA
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

# Transform the data to principal components
pca_data = pca.transform(scaled_data)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])

# Display the explained variance ratio and cumulative variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

print("Explained Variance Ratio:", explained_variance_ratio)
print("Cumulative Variance:", cumulative_variance)

# Visualize the data in PC1 and PC2 space
plt.figure(figsize=(10, 7))
plt.scatter(pca_data[:,0], pca_data[:,1], c=df['target'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of AAPL Data with Additional Features')
plt.grid(True)
plt.show()


loadings = pca.components_

loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=numerical_data.columns)

print(loadings_df)