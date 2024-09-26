# Step 1: Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Step 2: Import necessary libraries
import pandas as pd

# Step 3: Loading the dataset 
file_path = '/content/drive/My Drive/XAUUSD_Candlestick_4_Hour_BID_01.01.2020-25.05.2022.csv' # Update the path
data = pd.read_csv(file_path)

# Step 4: Displaying the first and last few rows of the dataset
print(data.head())
print(data.tail())


#Step 5: Removing rows where the 'volume' column is zero
data_cleaned = data[data['Volume'] != 0]  # Ensure 'volume' is the correct column name

# Displaying the cleaned data
print(data_cleaned.head())
data.tail()

#Step 6:Preprocessing
from sklearn.preprocessing import MinMaxScaler

# 1. Converting 'Gmt time' to datetime format with dayfirst=True
data['Gmt time'] = pd.to_datetime(data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f', dayfirst=True)

# 2. Setting 'Gmt time' as the index
data.set_index('Gmt time', inplace=True)

# 3. Sorting data by index 
data.sort_index(inplace=True)

# 4. Drop any rows with missing values (if necessary)
data.dropna(inplace=True)

# 5. Normalize the numerical columns (Low, High, Open, Close, Volume)
scaler = MinMaxScaler()
data[['Low', 'High', 'Open', 'Close', 'Volume']] = scaler.fit_transform(data[['Low', 'High', 'Open', 'Close', 'Volume']])

# 7. Dropping any rows that may have NaN values after creating new features
data.dropna(inplace=True)

# Displaying the preprocessed data
print(data.head())
