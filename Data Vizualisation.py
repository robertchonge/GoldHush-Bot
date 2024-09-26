#importing libraries

import matplotlib.pyplot as plt
import seaborn as sns

# Setting the aesthetics for the plots
sns.set(style='whitegrid')

# 1. Plotting the closing price over time
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.title('XAUUSD Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# 2. Plotting volume over time
plt.figure(figsize=(14, 7))
plt.bar(data.index, data['Volume'], color='orange')
plt.title('Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

# 3. Plotting High and Low prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['High'], label='High Price', color='green')
plt.plot(data.index, data['Low'], label='Low Price', color='red')
plt.title('XAUUSD High and Low Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# 4. Calculating and Ploting Returns
# Calculate returns using pct_change()
data['Returns'] = data['Close'].pct_change()

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Returns'], label='Returns', color='purple')
plt.title('XAUUSD Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)  # Horizontal line at 0
plt.legend()
plt.show()

# 5. Plotting  Moving Averages
# Calculate 5-period and 10-period moving averages
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_10'] = data['Close'].rolling(window=10).mean()

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['MA_5'], label='5-Period MA', color='orange')
plt.plot(data.index, data['MA_10'], label='10-Period MA', color='green')
plt.title('XAUUSD Close Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()