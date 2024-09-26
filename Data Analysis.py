# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Step 2: Import necessary libraries
import pandas as pd

# Step 3: Load the dataset (replace 'path_to_your_file.csv' with your actual file path)
file_path = '/content/drive/My Drive/XAUUSD_Candlestick_4_Hour_BID_01.01.2020-25.05.2022.csv' # Update the path
data = pd.read_csv(file_path)

# Step 4: Display the first few rows of the dataset
print(data.head())