#installing ta library
pip install ta

#required libraries
import pandas as pd
import numpy as np
from ta.trend import WMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import ForceIndexIndicator
import websocket
import json
import threading
import requests
import joblib
from flask import Flask, jsonify

# Flask App Initialization
app = Flask(__name__)

# Loading the trained models
linear_model = joblib.load('linear_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Trading parameters
LOT_SIZE = 0.05
SL_PIPS = 50
TP_PIPS = 50
AUTO_TRADING = False  # Start in manual mode
DERIV_API_URL = "https://api.deriv.com/v1"  # Base URL for Deriv API

# Deriv API credentials
API_KEY = "zS35ST8M5PMuDO8"
API_SECRET = "zS35ST8M5PMuDO8"

# Global variables to store trading data and stats
last_signal = None
last_price = None
account_balance = 1000  # Start with a default balance of 1000
win_loss_ratio = 0  # Initial win-loss ratio
pips_gained = 0  # Initial pips gained
total_trades = 0
winning_trades = 0
signal_probability = 0  # This will be calculated dynamically

# Function for calculating indicators
def calculate_indicators(data):
    data['WMA_95'] = WMAIndicator(data['Close'], window=95).wma()
    macd = MACD(data['Close'], window_slow=25, window_fast=13, window_sign=10)
    data['MACD_Line'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['RSI_14'] = RSIIndicator(data['Close'], window=14).rsi()
    data['Force_Index'] = ForceIndexIndicator(data['Close'], data['Volume'], window=13).force_index()
    return data

# Function to detect price action patterns
def detect_price_action_patterns(data):
    # Initialize a new column for patterns
    data['Pattern'] = "No Pattern"

    if len(data) < 3:
        return data  # Not enough data to identify patterns

    # Detect Bullish Pin Bar
    if (
        data['Close'].iloc[-1] > data['Open'].iloc[-1] and
        (data['Close'].iloc[-1] - data['Open'].iloc[-1]) < 0.2 * (data['High'].iloc[-1] - data['Low'].iloc[-1]) and
        ((data['High'].iloc[-1] - data['Close'].iloc[-1]) > 2 * (data['Close'].iloc[-1] - data['Open'].iloc[-1]))
    ):
        data['Pattern'].iloc[-1] = 'Bullish Pin Bar'

    # Detect Bearish Pin Bar
    elif (
        data['Close'].iloc[-1] < data['Open'].iloc[-1] and
        (data['Open'].iloc[-1] - data['Close'].iloc[-1]) < 0.2 * (data['High'].iloc[-1] - data['Low'].iloc[-1]) and
        ((data['Open'].iloc[-1] - data['Low'].iloc[-1]) > 2 * (data['Open'].iloc[-1] - data['Close'].iloc[-1]))
    ):
        data['Pattern'].iloc[-1] = 'Bearish Pin Bar'

    # Detect Bullish Engulfing
    elif (
        data['Close'].iloc[-1] > data['Open'].iloc[-1] and
        data['Open'].iloc[-2] > data['Close'].iloc[-2] and
        data['Close'].iloc[-1] > data['Open'].iloc[-2] and
        data['Open'].iloc[-1] < data['Close'].iloc[-2]
    ):
        data['Pattern'].iloc[-1] = 'Bullish Engulfing'

    # Detect Bearish Engulfing
    elif (
        data['Close'].iloc[-1] < data['Open'].iloc[-1] and
        data['Open'].iloc[-2] < data['Close'].iloc[-2] and
        data['Close'].iloc[-1] < data['Open'].iloc[-2] and
        data['Open'].iloc[-1] > data['Close'].iloc[-2]
    ):
        data['Pattern'].iloc[-1] = 'Bearish Engulfing'

    # Detect Inside Bar
    elif (
        data['High'].iloc[-1] < data['High'].iloc[-2] and
        data['Low'].iloc[-1] > data['Low'].iloc[-2]
    ):
        data['Pattern'].iloc[-1] = 'Inside Bar'

    return data

# Predict price using the models
def predict_price(data):
    features = data[['WMA_95', 'MACD_Line', 'RSI_14', 'Force_Index']]

    # Ensuring the input is reshaped properly
    features_array = features.values.reshape(1, -1)

    # Predicting with all three models and take an average of their predictions
    linear_pred = linear_model.predict(features_array)
    gb_pred = gb_model.predict(features_array)
    rf_pred = rf_model.predict(features_array)

    avg_pred = np.mean([linear_pred, gb_pred, rf_pred])
    return avg_pred[0]

# Placing  a trade using Deriv API
def place_trade(action, last_price):
    global account_balance, total_trades, winning_trades, pips_gained

    print(f"Placing {action} trade with lot size {LOT_SIZE}, SL: {SL_PIPS}, TP: {TP_PIPS}")

    sl_price = last_price - (SL_PIPS * 0.01) if action == 'buy' else last_price + (SL_PIPS * 0.01)
    tp_price = last_price + (TP_PIPS * 0.01) if action == 'buy' else last_price - (TP_PIPS * 0.01)

    payload = {
        "action": action,
        "amount": LOT_SIZE,
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "symbol": "XAUUSD",
        "api_key": API_KEY,
        "api_secret": API_SECRET
    }

    response = requests.post(f"{DERIV_API_URL}/trade", json=payload)
    total_trades += 1  # Increment the total trade count
    if response.status_code == 200:
        print(f"Trade successful: {response.json()}")
        account_balance += 25  # Increment account balance for successful trade
        winning_trades += 1  # Increment win count
        pips_gained += TP_PIPS  # Add the pips gained
    else:
        print(f"Trade failed: {response.json()}")
        account_balance -= 25  # Decrement account balance for failed trade
        pips_gained -= SL_PIPS  # Subtract pips lost

    # Update the win-loss ratio
    if total_trades > 0:
        global win_loss_ratio
        win_loss_ratio = (winning_trades / total_trades) * 100

# Function to toggle automatic trading
def toggle_auto_trading():
    global AUTO_TRADING
    AUTO_TRADING = not AUTO_TRADING
    print(f"Automatic trading is now {'enabled' if AUTO_TRADING else 'disabled'}")

# Main trading logic
def trading_logic(signal, last_price):
    if AUTO_TRADING:
        if signal == 'Buy':
            place_trade('buy', last_price)
        elif signal == 'Sell':
            place_trade('sell', last_price)
    else:
        print("Manual trading mode: No action taken.")

# Strategy to generate trading signals
def generate_signal(df):
    last_row = df.iloc[-1]

    # Predict price using the models
    predicted_price = predict_price(df)

    # Long Entry Logic
    if (
        (last_row['Pattern'] in ['Bullish Pin Bar', 'Inside Bar', 'Bullish Engulfing']) and
        last_row['Close'] > last_row['WMA_95'] and
        last_row['MACD_Line'] > last_row['MACD_Signal'] and
        last_row['RSI_14'] < 30 and
        predicted_price > last_row['Close']  # Prediction support for Buy
    ):
        return 'Buy'

    # Short Entry Logic
    if (
        (last_row['Pattern'] in ['Bearish Pin Bar', 'Inside Bar', 'Bearish Engulfing']) and
        last_row['Close'] < last_row['WMA_95'] and
        last_row['MACD_Line'] < last_row['MACD_Signal'] and
        last_row['RSI_14'] > 70 and
        predicted_price < last_row['Close']  # Prediction support for Sell
    ):
        return 'Sell'

    return None

# WebSocket message handler
def on_message(ws, message):
    global last_signal, last_price
    data = json.loads(message)

    if 'tick' in data:
        tick_data = data['tick']
        last_price = tick_data.get('quote')
        new_data = {
            'Low': tick_data.get('bid'),
'High': tick_data.get('ask'),
            'Close': last_price,
            'Open': tick_data.get('open'),
            'Volume': tick_data.get('volume'),
            'Time': tick_data.get('epoch')
        }

        # Convert tick data to DataFrame for easier manipulation
        df = pd.DataFrame([new_data])

        # Calculate indicators and detect patterns
        df = calculate_indicators(df)
        df = detect_price_action_patterns(df)

        # Generate trading signal based on the strategy
        signal = generate_signal(df)

        if signal and signal != last_signal:
            print(f"New Signal: {signal}")
            last_signal = signal

            # Execute trading logic based on the signal
            trading_logic(signal, last_price)

        else:
            print("No new signal. Holding position.")

# WebSocket connection handler
def on_open(ws):
    print("WebSocket connection opened.")
    ws.send(json.dumps({
        "ticks": "XAUUSD",
        "subscribe": 10,
        "api_key": API_KEY,
        "api_secret": API_SECRET
    }))

# WebSocket error handler
def on_error(ws, error):
    print(f"WebSocket error: {error}")

# WebSocket close handler
def on_close(ws):
    print("WebSocket connection closed.")

# WebSocket client thread to handle live market data
def websocket_client():
    ws = websocket.WebSocketApp("wss://ws.binaryws.com/websockets/v3?app_id=64298",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()

# Flask route to toggle automatic trading
@app.route('/toggle_auto_trading', methods=['POST'])
def toggle_auto():
    toggle_auto_trading()
    return jsonify({"message": f"Automatic trading {'enabled' if AUTO_TRADING else 'disabled'}"})

# Flask route to get the current stats
@app.route('/stats', methods=['GET'])
def get_stats():
    stats = {
        "account_balance": account_balance,
        "win_loss_ratio": win_loss_ratio,
        "pips_gained": pips_gained,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "signal_probability": signal_probability
    }
    return jsonify(stats)

# Function to calculate signal probability based on historical data
def update_signal_probability():
    global signal_probability
    if total_trades > 0:
        signal_probability = (winning_trades / total_trades) * 100

# Start WebSocket in a separate thread
def start_websocket_thread():
    websocket_thread = threading.Thread(target=websocket_client)
    websocket_thread.daemon = True
    websocket_thread.start()

if __name__ == '__main__':
    # Start WebSocket client for live data
    start_websocket_thread()

    # Start Flask server to provide trading stats and control auto trading
    app.run(host='0.0.0.0', port=5001)
