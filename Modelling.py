import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import gym  # Import OpenAI Gym for reinforcement learning

# Feature Engineering: Create lag features for the Close price
for lag in range(1, 6):
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

# Drop NaN values created by lagging
data.dropna(inplace=True)

# Define features and target variable
features = ['Low', 'High', 'Open', 'Volume'] + [f'Close_Lag_{lag}' for lag in range(1, 6)]
target = 'Close'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)

# Evaluate Linear Regression
print("Linear Regression Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print("R^2 Score:", r2_score(y_test, y_pred_linear))

# 2. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate Random Forest
print("\nRandom Forest Regressor Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R^2 Score:", r2_score(y_test, y_pred_rf))

# 3. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

# Evaluate Gradient Boosting
print("\nGradient Boosting Regressor Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_gb))
print("R^2 Score:", r2_score(y_test, y_pred_gb))

# Reinforcement Learning Environment Setup
class TradingEnvironment(gym.Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(features),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step][features].values

    def step(self, action):
        # Execute the action and calculate the next state and reward
        current_price = self.data.iloc[self.current_step]['Close']

        if action == 0:  # Buy
            reward = self.data.iloc[self.current_step + 1]['Close'] - current_price
        elif action == 1:  # Sell
            reward = current_price - self.data.iloc[self.current_step + 1]['Close']
        else:  # Hold
            reward = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self.data.iloc[self.current_step][features].values if not done else None

        return next_state, reward, done, {}
# Q-learning agent
class QLearningAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.q_table = np.zeros((len(data), action_size))  # Initialize Q-table
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return np.random.choice(self.action_size)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        # Convert next_state to an integer index if it's not None
        next_state_idx = int(next_state[0]) if next_state is not None else 0
        best_next_action = np.argmax(self.q_table[next_state_idx]) if next_state is not None else 0
        target = reward + self.discount_factor * self.q_table[next_state_idx][best_next_action] if next_state is not None else reward
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

# Training the Reinforcement Learning Agent
env = TradingEnvironment(data)
agent = QLearningAgent(action_size=env.action_space.n)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(env.current_step)
        next_state, reward, done, _ = env.step(action)
        agent.learn(env.current_step, action, reward, next_state)
        state = next_state

print("Training completed.")