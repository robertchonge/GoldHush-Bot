# Step 1: Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Step 2: Import necessary libraries
import pandas as pd

# Step 3: Loading the dataset (replace 'path_to_your_file.csv' with your actual file path)
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

