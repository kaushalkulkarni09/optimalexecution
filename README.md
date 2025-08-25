Optimal Trade Execution Strategies

This project is a Python-based simulation environment that compares the performance of several financial strategies designed for executing a large stock order. It uses a Monte Carlo approach on historical market data to determine the most effective strategy for minimizing transaction costs and market impact.

Strategies Implemented

The project evaluates a mix of traditional and modern algorithms:

Time-Weighted Average Price (TWAP): A simple strategy that divides a large order into equal-sized chunks and executes them at regular time intervals to spread the trade out over time.

Volume-Weighted Average Price (VWAP): A strategy that attempts to trade in line with the historical volume profile of the asset, aiming to execute orders at the average price weighted by the day's trading volume.

Almgren-Chriss: A classic theoretical model that balances cost and risk.

Dynamic Programming: A numerical method for finding an optimal solution in a discretized state space.

Deep Reinforcement Learning (DRL):

DQN (Deep Q-Network)

A2C (Advantage Actor-Critic)

DDPG (Deep Deterministic Policy Gradient)
