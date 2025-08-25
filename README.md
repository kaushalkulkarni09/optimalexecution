### **Optimal Liquidation of Large Stock Orders: A Comparative Study of Algorithmic Trading Strategies**

This repository contains the codebase and simulation environment for project focused on the comparative performance of various algorithmic execution strategies for large-scale stock orders. The primary objective is to evaluate and contrast the efficacy of classic heuristic models, foundational theoretical frameworks, and state-of-the-art deep reinforcement learning agents in minimizing market impact and overall execution costs within a high-fidelity market simulator.

**Methodological Framework**

The study employs a robust Monte Carlo simulation methodology, utilizing historical high-frequency trade and quote data for the SPY ETF to construct a realistic and non-stationary market environment. The performance of each algorithm is measured based on a total cost function that aggregates both realized market impact costs and the opportunity costs associated with an adverse price drift over the execution horizon.

**Algorithmic Trading Frameworks**

This research investigates the following classes of algorithms, representing a spectrum from classic rule-based heuristics to sophisticated adaptive learning agents. 
* **Heuristic Benchmarks:**
 
  * **Time-Weighted Average Price (TWAP):** A rule-based strategy that liquidates an order linearly over a predefined time interval, making no adjustments for market volatility or liquidity. It serves as a baseline for measuring performance against more adaptive models.
  * **Volume-Weighted Average Price (VWAP):** A rule-based strategy designed to align an order's execution with the historical volume profile of the asset. The objective is to achieve a final average price that is commensurate with the day's volume-weighted average price.

* **Theoretical & Numerical Optimization Models:**

  * **Almgren-Chriss:** A seminal theoretical model for optimal liquidation. The algorithm minimizes the total expected execution cost by balancing permanent market impact, temporary market impact, and volatility risk. It assumes a linear market impact function and a Gaussian price process.
 
  * **Dynamic Programming (DP):** A numerical solution to the optimal execution problem, which is formulated as a stochastic control problem. The algorithm discretizes the state space (time, remaining inventory) and uses backward induction to determine the optimal trading policy at each step, accounting for all possible future market states.

* **Deep Reinforcement Learning (DRL) Agents:** These agents learn an optimal liquidation policy by interacting with the simulated market environment. The agents are trained to maximize a reward function that is inversely proportional to execution costs.

  * **Deep Q-Network (DQN):** A value-based DRL agent that approximates the optimal action-value function (Q-function) using a deep neural network. The agent's policy is to select the action with the highest estimated Q-value at each time step.
  *  **Advantage Actor-Critic (A2C):**  An on-policy actor-critic algorithm that leverages a dual network architecture. The actor network learns the policy (the action to take), while the critic network evaluates the value of the current state, guiding the actor's learning process.
  *  **Deep Deterministic Policy Gradient (DDPG):** An off-policy actor-critic algorithm designed for environments with continuous action spaces, allowing the agent to select a precise quantity of shares to trade. It combines a deterministic policy with a critic network to achieve stable learning.

**Simulation Results and Performance Analysis**

The empirical performance of each strategy was evaluated over a simulated period, with total execution cost serving as the primary metric. The results demonstrate the superior performance of the DRL agent, particularly the DQN model, in mitigating market impact and minimizing liquidation costs.

| Strategy | Mean Total Execution Cost | Cost Standard Deviation |
|:---|:---:|:---:|
| Deep Reinforcement Learning (DQN) | $152,511.38 | $420,551.58 |
| Almgren-Chriss | $316,777.70 | $953,446.67 |
| TWAP | $443,039.11 | $1,053,331.33 |
| Deep Reinforcement Learning (DDPG) | $475,887.57 | $719,528.80 |
| Deep Reinforcement Learning (A2C) | $492,814.56 | $1,197,348.58 |
| VWAP (Linear Peak) | $716,347.09 | $1,099,661.04 |
| Dynamic Programming | $851,529.27 | $1,202,194.15 |

**Performance Analysis and Discussion**

The simulation results provide compelling evidence of the hegemonic performance of the deep reinforcement learning agent, specifically the DQN model. The DQN agent achieved a mean execution cost that was over 50% lower than the nearest competitor, the classic Almgren-Chriss model. This finding is significant as it demonstrates that a data-driven, adaptive learning approach can outperform well-established theoretical frameworks that rely on idealized assumptions about market behavior.

Furthermore, the DQN model's low standard deviation is a critical finding. A lower standard deviation indicates a more consistent and reliable performance across different market conditions, making it a more robust and predictable strategy for real-world application.

The relative performance of the other models also aligns with theoretical expectations. The Almgren-Chriss model, despite its underlying assumptions, provides a strong benchmark, outperforming the simpler TWAP and VWAP heuristics. The lower performance of the other DRL agents (DDPG and A2C) compared to DQN suggests that for this specific problem and simulated environment, the value-based learning approach of DQN was more effective at converging to the optimal policy. The dynamic programming approach, while theoretically sound, performed the worst, likely due to the computational challenges and approximation errors inherent in discretizing a high-dimensional state space.

In summary, these results validate the application of modern machine learning techniques, particularly DRL, as a superior paradigm for optimal trade execution in complex, stochastic market environments.

**Instructions for Reproducibility**

This project is a self-contained environment. To replicate the simulation and analysis, please follow the steps below.

**System Requirements**

The following Python libraries are required for the simulation environment:

* yfinance
* numpy
* pandas
* torch
* matplotlib
* seaborn

These dependencies can be installed via pip using the following command: **pip install yfinance numpy pandas torch matplotlib seaborn**

**Execution**

The primary simulation script, optimal_execution.py, will automatically download the necessary historical data, perform the comparative analysis, and generate the corresponding visualization plots. Execute the script from the terminal as follows: **python optimalexecution.py**.









