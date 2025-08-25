# Brief summary of the file:
# optimal_execution.py - Optimal Execution Strategies with Real-World Market Impact & Deep Reinforcement Learning
# It includes integrated Matplotlib visualizations and a DDPG agent for continuous action spaces.

import math
import random
import time
import logging
import sys
from collections import deque
import uuid

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical # For A2C action sampling

import matplotlib.pyplot as plt # Import for plotting
import matplotlib.ticker as mticker # For formatting axes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- CONFIGURATION AND GLOBAL PARAMETERS ---
TICKER = "SPY" 
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"

# Heston Model Parameters for market price and volatility evolution
HESTON_MU = 0.05 # Annualized drift for the underlying asset price
HESTON_VOL_INITIAL = 0.20 # Initial annualized volatility (sqrt of variance)
HESTON_KAPPA = 2.0 # Rate of mean reversion for volatility
HESTON_THETA = 0.04 # Long-term mean of volatility (variance)
HESTON_XI = 0.5 # Volatility of volatility (vol of vol) to amplify stochasticity
HESTON_RHO = -0.7 # Correlation between price and volatility Wiener processes

TOTAL_SHARES_TO_TRADE = 10000.0
T_HORIZON_DAYS = 5 # Trading horizon in number of intervals (days)

GAMMA_PERMANENT_IMPACT = 1e-6 # Coefficient for permanent market impact
BETA_TEMPORARY_IMPACT = 5e-6 # Coefficient for temporary market impact

BID_ASK_SPREAD_BPS = 5 # Base bid-ask spread in basis points (5 means 0.05%)
SPREAD_VOLATILITY_FACTOR = 100 # How much spread increases with 1 unit of annualized volatility (0.2 -> 0.05% * 100 = 5 bps extra)

OPPORTUNITY_COST_PER_DAY_FACTOR = 0.0001 # 0.01% of notional value per day (cost for holding unliquidated inventory)

# NEW: Order Sizing Constraints
MIN_SHARES_PER_SLICE = 100.0 # Minimum shares per trade if a trade is initiated
MAX_SHARES_PER_SLICE_FACTOR = 0.5 # Max shares per trade as a factor of TOTAL_SHARES_TO_TRADE (e.g., 0.5 means max 5000 shares for 10000 total)

NUM_MONTE_CARLO_PATHS = 50 # Number of simulation paths for each strategy
HISTORICAL_WINDOW_SIZE = 250 # Size of window for random historical data segments
NUM_SAMPLE_VIZ_PATHS = 2 # Number of simulation paths to store detailed history for visualization

# DRL (DQN) Parameters
DQN_LEARNING_RATE = 1e-4
DQN_DISCOUNT_FACTOR = 0.99
DQN_EXPLORATION_RATE_START = 1.0
DQN_EXPLORATION_RATE_END = 0.01
DQN_EXPLORATION_DECAY = 0.005
DQN_EPISODES = 1000 # Number of training episodes for DQN
DQN_BATCH_SIZE = 64
DQN_REPLAY_BUFFER_SIZE = 10000
DQN_TARGET_UPDATE_FREQ = 10
DQN_GRADIENT_CLIP = 1.0

# A2C DRL Parameters
A2C_ACTOR_LR = 1e-5 # Learning rate for the Actor network
A2C_CRITIC_LR = 1e-4 # Learning rate for the Critic network
A2C_DISCOUNT_FACTOR = 0.99
A2C_ENTROPY_BETA = 0.001 # Coefficient for entropy regularization (encourages exploration)
A2C_EPISODES = 1000 # Number of episodes for A2C training

# DDPG DRL Parameters (New for Continuous Action Space)
DDPG_ACTOR_LR = 1e-5 # Learning rate for the Actor network (policy)
DDPG_CRITIC_LR = 1e-4 # Learning rate for the Critic network (Q-value)
DDPG_DISCOUNT_FACTOR = 0.99
DDPG_TAU = 0.005 # Soft update rate for target networks
DDPG_BATCH_SIZE = 64
DDPG_REPLAY_BUFFER_SIZE = 10000
DDPG_EPISODES = 1000 # Number of training episodes
DDPG_GRADIENT_CLIP = 1.0
# Ornstein-Uhlenbeck noise parameters for exploration
DDPG_OU_MU = 0.0
DDPG_OU_THETA = 0.15 # Rate of mean reversion
DDPG_OU_SIGMA = 0.5 # **Increased for more exploration**
DDPG_OU_DT = 1.0 # Time step for OU noise (daily)


# Discretization buckets for DP and discrete RL state/action spaces
DP_RL_INVENTORY_BUCKETS = 20
DP_RL_ACTION_BUCKETS = 10 # Used for DQN and A2C
DP_RL_PRICE_BUCKETS = 10
DP_RL_VOL_BUCKETS = 10

# --- REWARD ADJUSTMENT ---
# CRITICAL ADJUSTMENT: Reduce the terminal penalty to avoid training instability
TERMINAL_INVENTORY_PENALTY_FACTOR = 5.0 # Reduced from 1e3 to a smaller value, as it will be scaled by REWARD_SCALE_BASE

# Placeholder for reward scaling factor - will be calculated from historical data
REWARD_SCALE_BASE = 1.0

# --- DATA LOADING AND PREPROCESSING ---
def load_historical_data(ticker, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance and calculates
    daily returns, logs those returns, and annualized daily volatility.
    """
    logger.info(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            logger.error(f"No data for {ticker}. Check ticker/date.")
            return None

        # Adjust column name based on yfinance's auto_adjust behavior
        price_col = 'Adj Close'
        if price_col not in data.columns:
            price_col = 'Close'
            logger.info(f"Using '{price_col}' column for prices as 'Adj Close' was not found.")

        if price_col not in data.columns or 'Volume' not in data.columns:
            logger.error(f"Missing '{price_col}' or 'Volume' in downloaded data for {ticker}.")
            return None

        data['Returns'] = data[price_col].pct_change()
        data['LogReturns'] = np.log(data[price_col] / data[price_col].shift(1))

        data['DailyVolatility'] = data['LogReturns'].rolling(window=20).std() * np.sqrt(252)
        data = data.dropna()
        logger.info(f"Downloaded {len(data)} data points for {ticker}.")
        return data[[price_col, 'Volume', 'Returns', 'DailyVolatility']]
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None

# --- MATHEMATICAL MODELS ---
def heston_sde_step(S_t, v_t, mu, kappa, theta, xi, rho, dt, Z1, Z2):
    """
    Performs one step of the Heston stochastic differential equation for price (S) and variance (v).
    This function introduces the stochasticity in price and volatility evolution.
    S_t: Current mid-price
    v_t: Current variance (volatility squared)
    mu: Daily drift for the underlying asset (annualized_mu / 252)
    kappa: Rate of mean reversion for volatility
    theta: Long-term mean of volatility (variance)
    xi: Volatility of volatility (vol of vol) to amplify stochasticity
    rho: Correlation between price and volatility Wiener processes
    dt: Time step (e.g., 1.0 for daily)
    Z1, Z2: Independent standard normal random variables
    """
    # Ensuring that the variance is non-negative
    v_t = max(v_t, 1e-8)

    # Correlated Wiener processes
    dW1 = Z1 * np.sqrt(dt) # Price Wiener process
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * Z2 * np.sqrt(dt) # Volatility Wiener process

    # Update variance (CIR process)
    dv = kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * dW2
    v_t_plus_dt = v_t + dv
    v_t_plus_dt = max(v_t_plus_dt, 1e-8) # Ensure variance remains non-negative

    # Update price (Geometric Brownian Motion with stochastic volatility)
    dS = mu * S_t * dt + np.sqrt(v_t_plus_dt) * S_t * dW1
    S_t_plus_dt = S_t + dS
    S_t_plus_dt = max(S_t_plus_dt, 1e-8) # Ensure price remains non-negative

    return S_t_plus_dt, v_t_plus_dt

def permanent_market_impact_linear(executed_shares, gamma_param, is_buy):
    """
    Calculates the permanent market impact on mid-price based on a linear model.
    The impact is proportional to the executed shares and applies to all subsequent prices.
    """
    sign = 1 if is_buy else -1
    return sign * gamma_param * executed_shares

def temporary_market_impact_quadratic_cost(executed_shares, beta_param):
    """
    Calculates the temporary market impact cost based on a quadratic model.
    This cost is incurred only for the specific trade and does not affect future prices.
    """
    return beta_param * (executed_shares**2)

# --- OPTIMAL EXECUTION STRATEGIES ---
class OptimalExecutionStrategy:
    """
    Base class for optimal execution strategies.
    Defines common attributes and an interface for getting a trading schedule.
    """
    def __init__(self, total_shares, T_horizon_intervals):
        if not (isinstance(total_shares, (int, float)) and total_shares > 0):
            raise ValueError("Total shares must be positive.")
        if not (isinstance(T_horizon_intervals, int) and T_horizon_intervals > 0):
            raise ValueError("Time horizon (intervals) must be positive.")
        self.total_shares = float(total_shares)
        self.T_horizon_intervals = int(T_horizon_intervals)
        self.trading_schedule = np.zeros(self.T_horizon_intervals)

    def get_trading_schedule(self, *args, **kwargs):
        """
        Abstract method to be implemented by concrete strategy classes.
        Returns a numpy array representing the shares to trade in each interval.
        """
        raise NotImplementedError

class TWAPStrategy(OptimalExecutionStrategy):
    """
    Time-Weighted Average Price (TWAP) strategy.
    Executes an equal amount of shares in each time interval.
    """
    def __init__(self, total_shares, T_horizon_intervals):
        super().__init__(total_shares, T_horizon_intervals)
        self._calculate_schedule()

    def _calculate_schedule(self):
        """Calculates the fixed TWAP schedule."""
        self.shares_per_interval = self.total_shares / self.T_horizon_intervals
        self.trading_schedule.fill(self.shares_per_interval)

    def get_trading_schedule(self, *args, **kwargs):
        """Returns the pre-calculated TWAP trading schedule."""
        return self.trading_schedule

class VWAPStrategy(OptimalExecutionStrategy):
    """
    Volume-Weighted Average Price (VWAP) strategy.
    Attempts to match a predefined volume profile over the trading horizon.
    """
    def __init__(self, total_shares, T_horizon_intervals, volume_profile_type='linear_peak_middle'):
        super().__init__(total_shares, T_horizon_intervals)
        self.volume_profile_type = volume_profile_type
        self._calculate_schedule()

    def _calculate_schedule(self):
        """
        Calculates the VWAP trading schedule based on a specified volume profile.
        """
        if self.volume_profile_type == 'uniform':
            self.trading_schedule.fill(self.total_shares / self.T_horizon_intervals)
        elif self.volume_profile_type == 'linear_peak_middle':
            t_points = np.arange(self.T_horizon_intervals)
            if self.T_horizon_intervals % 2 == 0:
                half_len = self.T_horizon_intervals // 2
                profile = np.concatenate([np.arange(1, half_len + 1), np.arange(half_len, 0, -1)])
            else:
                half_len = self.T_horizon_intervals // 2
                profile = np.concatenate([np.arange(1, half_len + 1), [half_len + 1], np.arange(half_len, 0, -1)])

            # Normalize profile to ensure total number shares are traded
            if np.sum(profile) > 0:
                self.trading_schedule = (profile / np.sum(profile)) * self.total_shares
            else:
                self.trading_schedule.fill(self.total_shares / self.T_horizon_intervals)
        elif self.volume_profile_type == 'linear_increasing':
            t_points = np.arange(self.T_horizon_intervals)
            raw_profile = t_points + 1 # Simple linear increase
            if np.sum(raw_profile) > 0:
                self.trading_schedule = (raw_profile / np.sum(raw_profile)) * self.total_shares
            else:
                self.trading_schedule.fill(self.total_shares / self.T_horizon_intervals)
        else:
            raise ValueError(f"Unknown volume_profile_type: {self.volume_profile_type}")

    def get_trading_schedule(self, *args, **kwargs):
        """Returns the pre-calculated VWAP trading schedule."""
        return self.trading_schedule

class AlmgrenChrissStrategy(OptimalExecutionStrategy):
    """
    Implements the Almgren-Chriss optimal execution strategy.
    Calculates a trading schedule that balances market impact cost and price volatility risk.
    """
    def __init__(self, total_shares, T_horizon_intervals,
                 sigma_daily, risk_aversion, gamma_ac, eta_ac):
        super().__init__(total_shares, T_horizon_intervals)
        self.sigma_daily = sigma_daily # Daily volatility of the underlying asset
        self.risk_aversion = risk_aversion # Investor's aversion to risk
        self.gamma_ac = gamma_ac # Permanent market impact coefficient for AC model
        self.eta_ac = eta_ac # Temporary market impact coefficient for AC model
        self._calculate_schedule()

    def _calculate_schedule(self):
        """
        Calculates the Almgren-Chriss trading schedule.
        Handles edge cases where parameters might lead to undefined or zero schedules.
        """

        if self.eta_ac <= 1e-9 or self.risk_aversion <= 1e-9:
            self.trading_schedule.fill(self.total_shares / self.T_horizon_intervals)
            logger.warning("Almgren-Chriss: Risk aversion or temporary impact is effectively zero. Defaulting to TWAP.")
            return

        phi = self.risk_aversion * (self.sigma_daily**2) # Risk component
        if 2 * self.eta_ac <= 0:
            raise ValueError("Almgren-Chriss: Temporary impact coeff must be positive for kappa calculation.")
        kappa = np.sqrt(phi / (2 * self.eta_ac))
        total_time_duration = float(self.T_horizon_intervals)


        if kappa * total_time_duration < 1e-6: # Check if kappa is effectively zero relative to horizon
            self.trading_schedule.fill(self.total_shares / self.T_horizon_intervals)
            logger.warning("Almgren-Chriss: Kappa is too small, resulting in near-zero sinh. Defaulting to TWAP.")
            return

        sinh_kappa_T_total = np.sinh(kappa * total_time_duration)

        # Calculate inventory levels (x_k) at each time step k
        x_k_values = np.zeros(self.T_horizon_intervals + 1)
        x_k_values[0] = self.total_shares # Starting inventory

        for k in range(self.T_horizon_intervals + 1):
            time_remaining = total_time_duration - k # T-k
            x_k_values[k] = self.total_shares * (np.sinh(kappa * time_remaining) / sinh_kappa_T_total)

        # Ensure inventory is non-negative and monotonically decreasing (cannot increase shares or go below zero)
        x_k_values = np.clip(x_k_values, 0, self.total_shares)
        for i in range(1, len(x_k_values)):
            if x_k_values[i] > x_k_values[i-1]:
                x_k_values[i] = x_k_values[i-1] # Ensure non-increasing inventory

        # Shares traded in each interval is the difference in inventory (x_k - x_{k+1})
        self.trading_schedule = np.diff(x_k_values) * -1 # Multiply by -1 because np.diff gives x_{k+1}-x_k
        self.trading_schedule[self.trading_schedule < 0] = 0.0 # Ensure no negative trades

        # Re-normalize the schedule to ensure total_shares are traded, accounting for potential clipping errors
        sum_scheduled = np.sum(self.trading_schedule)
        if sum_scheduled > 1e-9: # Avoid division by zero if schedule is all zeros
            self.trading_schedule = (self.trading_schedule / sum_scheduled) * self.total_shares
        else:
             logger.warning(f"Almgren-Chriss strategy produced a zero trading schedule (after re-normalization).")

    def get_trading_schedule(self, *args, **kwargs):
        """Returns the pre-calculated Almgren-Chriss trading schedule."""
        return self.trading_schedule

class OptimalExecutionDynamicProgramming(OptimalExecutionStrategy):
    """
    Implements an optimal execution strategy using Dynamic Programming.
    Discretizes the state space (inventory, mid-price, volatility) and action space (shares to trade)
    to find the optimal policy by minimizing expected future costs.
    """
    def __init__(self, total_shares, T_horizon_intervals,
                 gamma_permanent, beta_temporary,
                 mu, kappa_heston, theta_heston, xi_heston, rho_heston,
                 price_buckets, inventory_buckets, action_buckets, vol_buckets,
                 bid_ask_spread_bps, spread_volatility_factor,
                 opportunity_cost_per_day_factor): # Corrected kwarg name
        super().__init__(total_shares, T_horizon_intervals)
        self.gamma_permanent = gamma_permanent
        self.beta_temporary = beta_temporary
        self.mu = mu # This mu is now expected to be daily drift
        self.kappa_heston = kappa_heston
        self.theta_heston = theta_heston
        self.xi_heston = xi_heston
        self.rho_heston = rho_heston
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.spread_volatility_factor = spread_volatility_factor
        self.opportunity_cost_per_day_factor = opportunity_cost_per_day_factor


        self.price_buckets = price_buckets
        self.inventory_buckets = inventory_buckets
        self.action_buckets = action_buckets
        self.vol_buckets = vol_buckets

        # Discretized state and action spaces
        self.inventory_levels = np.linspace(0, total_shares, inventory_buckets)
        # Actions are shares to trade in an interval, up to twice the average daily trade
        self.actions = np.linspace(0, total_shares / T_horizon_intervals * 2, action_buckets)
        # Quantize actions to be multiples of a base unit if desired, here just rounding to common scale
        self.actions = np.round(self.actions / (total_shares / T_horizon_intervals / action_buckets)) * \
                       (total_shares / T_horizon_intervals / action_buckets)
        self.actions = self.actions[self.actions >= 0] # Ensure non-negative actions

        # Value function V(t, inventory, mid-price, vol) and optimal policy pi(t, inventory, mid-price, vol)
        self.value_function = np.zeros((T_horizon_intervals + 1, inventory_buckets, price_buckets, vol_buckets))
        self.optimal_policy = np.zeros((T_horizon_intervals, inventory_buckets, price_buckets, vol_buckets), dtype=int)

        self.price_levels = None # Will be set based on historical data range
        self.vol_levels = None   # Will be set based on historical data range

    def set_market_ranges(self, price_min, price_max, vol_min, vol_max):
        """Sets the ranges for price and volatility discretization."""
        self.price_levels = np.linspace(price_min, price_max, self.price_buckets)
        self.vol_levels = np.linspace(vol_min, vol_max, self.vol_buckets)
        self.vol_levels[self.vol_levels < 1e-8] = 1e-8 # Prevent zero or negative volatility

    def _get_inventory_idx(self, current_inventory):
        """Maps continuous inventory to its nearest discrete index."""
        return np.argmin(np.abs(self.inventory_levels - current_inventory))

    def _get_price_idx(self, current_price):
        """Maps continuous price to its nearest discrete index."""
        if self.price_levels is None: raise ValueError("Price levels not set for DP.")
        idx = np.argmin(np.abs(self.price_levels - current_price))
        return np.clip(idx, 0, self.price_buckets - 1) # Ensure index is within bounds

    def _get_vol_idx(self, current_vol):
        """Maps continuous volatility to its nearest discrete index."""
        if self.vol_levels is None: raise ValueError("Vol levels not set for DP.")
        idx = np.argmin(np.abs(self.vol_levels - current_vol))
        return np.clip(idx, 0, self.vol_buckets - 1) # Ensure index is within bounds

    def _calculate_execution_price_and_cost(self, mid_price, current_vol_variance, shares_executed, is_buy):
        """
        Calculates the effective execution price and the temporary market impact cost
        including the bid-ask spread (now dynamic based on volatility).
        """

        current_annualized_vol_std = np.sqrt(max(1e-8, current_vol_variance))
        dynamic_spread_bps = self.bid_ask_spread_bps + (current_annualized_vol_std * self.spread_volatility_factor)
        dynamic_spread_bps = max(float(self.bid_ask_spread_bps), dynamic_spread_bps) # Ensure it's at least the base spread
        dynamic_spread_bps_decimal = dynamic_spread_bps / 10000.0

        half_spread = mid_price * dynamic_spread_bps_decimal / 2.0

        effective_execution_price = mid_price # Default if no shares executed

        if shares_executed > 0:
            if is_buy:
                effective_execution_price = mid_price + half_spread
            else:
                effective_execution_price = mid_price - half_spread


            temp_impact_cost = temporary_market_impact_quadratic_cost(shares_executed, self.beta_temporary)


            if is_buy: # Buying, temporary impact increases price
                effective_execution_price += (temp_impact_cost / shares_executed)
            else: # Selling, temporary impact decreases price received
                effective_execution_price -= (temp_impact_cost / shares_executed)

            # Ensure effective execution price remains positive
            effective_execution_price = max(effective_execution_price, 1e-8)


            total_instantaneous_cost_component = (mid_price - effective_execution_price) * shares_executed
        else: # No shares executed, so no direct transaction costs this interval
            total_instantaneous_cost_component = 0.0

        return effective_execution_price, total_instantaneous_cost_component


    def _solve_dp(self):
        """
        Solves the dynamic programming problem using backward induction.
        Iterates from the last time step back to the first, calculating the optimal value
        function and policy for each state.
        """
        logger.info(f"Solving DP for Optimal Execution (Inv Buckets: {self.inventory_buckets}, Price Buckets: {self.price_buckets}, Vol Buckets: {self.vol_buckets})...")
        if self.price_levels is None or self.vol_levels is None:
            logger.error("DP market ranges not set. Cannot solve DP.")
            return

        # Terminal condition: At the last time step (T_horizon_intervals), if inventory remains, incur a penalty.
        for inv_idx in range(self.inventory_buckets):
            for p_idx in range(self.price_buckets):
                for v_idx in range(self.vol_buckets):
                    remaining_inv = self.inventory_levels[inv_idx]
                    if remaining_inv > 1e-9: # If there's remaining inventory
                        # Penalty is based on notional value at current mid-price
                        mid_price_at_terminal = self.price_levels[p_idx]
                        self.value_function[self.T_horizon_intervals][inv_idx][p_idx][v_idx] = TERMINAL_INVENTORY_PENALTY_FACTOR * remaining_inv * mid_price_at_terminal
                    else:
                        self.value_function[self.T_horizon_intervals][inv_idx][p_idx][v_idx] = 0.0 # No penalty if all liquidated


        for k in range(self.T_horizon_intervals - 1, -1, -1):
            for inv_idx in range(self.inventory_buckets):
                current_inventory = self.inventory_levels[inv_idx]


                if current_inventory < 1e-9:
                    for p_idx in range(self.price_buckets):
                        for v_idx in range(self.vol_buckets):
                            self.value_function[k][inv_idx][p_idx][v_idx] = 0.0
                            self.optimal_policy[k][inv_idx][p_idx][v_idx] = 0
                    continue

                for p_idx in range(self.price_buckets):
                    current_mid_price = self.price_levels[p_idx]
                    for v_idx in range(self.vol_buckets):
                        current_vol = self.vol_levels[v_idx] # This is variance
                        min_expected_total_cost = float('inf') # Initialize with a very high cost
                        best_action_idx = 0

                        # Iterate over all possible actions (shares to trade)
                        for action_idx, shares_to_trade in enumerate(self.actions):
                            # Cannot trade more shares than currently held
                            if shares_to_trade > current_inventory + 1e-9:
                                continue

                            # Calculate instantaneous cost (from this interval's trade)
                            # The effective_exec_price_this_interval is primarily for the simulation loop in run_simulation

                            _, instantaneous_trade_cost = \
                                self._calculate_execution_price_and_cost(current_mid_price, current_vol, shares_to_trade, is_buy=False)

                            next_inventory = current_inventory - shares_to_trade
                            next_inventory = max(0.0, next_inventory) # Inventory cannot be negative
                            next_inv_idx = self._get_inventory_idx(next_inventory)

                            # Calculate expected future cost by simulating multiple paths for price and vol
                            num_sample_paths_for_expectation = 5 # Monte Carlo sampling for expectation
                            expected_future_value_sum = 0.0 # Expected Value for inventory, NOT cost
                            for _ in range(num_sample_paths_for_expectation):
                                Z1_sim = np.random.randn()
                                Z2_sim = np.random.randn()

                                # Simulate mid-price after permanent impact from current trade, then apply Heston dynamics
                                sim_mid_price_after_impact = current_mid_price - permanent_market_impact_linear(shares_to_trade, self.gamma_permanent, is_buy=False)
                                sim_mid_price_next, sim_vol_next = heston_sde_step(
                                    sim_mid_price_after_impact, current_vol, self.mu, self.kappa_heston,
                                    self.theta_heston, self.xi_heston, self.rho_heston,
                                    1.0, Z1_sim, Z2_sim
                                )
                                # Map simulated continuous values to discrete indices
                                next_p_idx = self._get_price_idx(sim_mid_price_next)
                                next_v_idx = self._get_vol_idx(sim_vol_next)

                                # Add future cost from the next state (from value function at k+1)
                                future_cost_from_value_function = self.value_function[k + 1][next_inv_idx][next_p_idx][next_v_idx]

                                # Add opportunity cost for remaining inventory in the NEXT period
                                opportunity_cost_next_step = self.opportunity_cost_per_day_factor * next_inventory * sim_mid_price_next

                                expected_future_value_sum += (future_cost_from_value_function + opportunity_cost_next_step)

                            expected_future_cost_total = expected_future_value_sum / num_sample_paths_for_expectation

                            # Total cost for this action and path (instantaneous trade cost + expected future costs)
                            total_path_cost = instantaneous_trade_cost + expected_future_cost_total

                            if total_path_cost < min_expected_total_cost:
                                min_expected_total_cost = total_path_cost
                                best_action_idx = action_idx

                        self.value_function[k][inv_idx][p_idx][v_idx] = min_expected_total_cost
                        self.optimal_policy[k][inv_idx][p_idx][v_idx] = best_action_idx
        logger.info("DP solution complete.")

    def get_trading_schedule(self, initial_mid_price, initial_vol):
        """
        Derives the trading schedule by following the optimal policy forward in time,
        starting from the initial state.
        """
        schedule = np.zeros(self.T_horizon_intervals)
        current_inventory = self.total_shares
        current_mid_price = initial_mid_price
        current_vol = initial_vol

        for k in range(self.T_horizon_intervals):
            inv_idx = self._get_inventory_idx(current_inventory)
            p_idx = self._get_price_idx(current_mid_price)
            v_idx = self._get_vol_idx(current_vol)

            # If all shares are liquidated, stop trading
            if current_inventory < 1e-9:
                schedule[k:] = 0.0
                break

            # Get the optimal action from the pre-computed policy
            action_idx = self.optimal_policy[k][inv_idx][p_idx][v_idx]
            shares_to_trade = self.actions[action_idx]

            # Ensure shares traded don't exceed remaining inventory and are non-negative
            shares_to_trade = min(shares_to_trade, current_inventory)
            shares_to_trade = max(0.0, shares_to_trade)

            schedule[k] = shares_to_trade
            current_inventory -= shares_to_trade

            # Simulate the market for the next time step based on the trade

            price_impact_from_trade = permanent_market_impact_linear(shares_to_trade, self.gamma_permanent, is_buy=False)
            mid_price_after_permanent_impact = current_mid_price - price_impact_from_trade

            Z1_next = np.random.randn()
            Z2_next = np.random.randn()
            current_mid_price, current_vol = heston_sde_step(
                mid_price_after_permanent_impact, current_vol, self.mu, self.kappa_heston,
                self.theta_heston, self.xi_heston, self.rho_heston,
                1.0, Z1_next, Z2_next
            )

        total_scheduled_shares = np.sum(schedule)
        if total_scheduled_shares > 1e-9:
            schedule = (schedule / total_scheduled_shares) * self.total_shares
        schedule[schedule < 1e-9] = 0.0 # Final check for very small numbers
        return schedule

class ReplayBuffer:
    """
    A simple replay buffer for storing (state, action, reward, next_state, done) transitions.
    Used by DQN, A2C, and DDPG to break correlations in training data.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Adds a new transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Samples a random batch of transitions from the buffer."""
        if len(self.buffer) < batch_size:
            return None
        minibatch = random.sample(self.buffer, batch_size)

        return (torch.stack([s[0] for s in minibatch]).float(), # States
                torch.stack([s[1] for s in minibatch]).float() if isinstance(minibatch[0][1], torch.Tensor) and minibatch[0][1].dim() > 0 else torch.tensor([s[1] for s in minibatch], dtype=torch.long), # Actions (handle scalar vs tensor, force long for discrete)
                torch.tensor([s[2] for s in minibatch]).float(), # Rewards
                torch.stack([s[3] for s in minibatch]).float(), # Next States
                torch.tensor([s[4] for s in minibatch], dtype=torch.float32)) # Dones (explicitly float32)

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

class DQNAgent(nn.Module):
    """
    Deep Q-Network (DQN) architecture for estimating Q-values.
    A multi-layer perceptron (MLP) with ReLU activations.
    """
    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        # Deeper network with three hidden layers for better robustness
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64) # New hidden layer
        self.fc4 = nn.Linear(64, output_dim) # Output layer

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # Activation for the new layer
        return self.fc4(x) # Final output layer

class DeepReinforcementLearningAgent(OptimalExecutionStrategy):
    """
    Implements a Deep Q-Network (DQN) based optimal execution strategy.
    Learns an optimal trading policy by interacting with a simulated market environment.
    """
    def __init__(self, total_shares, T_horizon_intervals,
                 gamma_permanent, beta_temporary,
                 mu, kappa_heston, theta_heston, xi_heston, rho_heston,
                 inventory_buckets, action_buckets, # These are used to define action_space only, not for state buckets directly
                 learning_rate, discount_factor, exploration_start, exploration_end, exploration_decay,
                 batch_size, replay_buffer_size, target_update_freq, gradient_clip,
                 bid_ask_spread_bps, spread_volatility_factor,
                 opportunity_cost_per_day_factor,
                 reward_scale_base): # Added reward_scale_base
        super().__init__(total_shares, T_horizon_intervals)
        self.gamma_permanent = gamma_permanent
        self.beta_temporary = beta_temporary
        self.mu = mu # This mu is now expected to be daily drift
        self.kappa_heston = kappa_heston
        self.theta_heston = theta_heston
        self.xi_heston = xi_heston
        self.rho_heston = rho_heston
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.spread_volatility_factor = spread_volatility_factor
        self.opportunity_cost_per_day_factor = opportunity_cost_per_day_factor
        self.reward_scale_base = reward_scale_base # Store reward scaling factor


        self.inventory_levels = np.linspace(0, total_shares, inventory_buckets) # Used to define action space indirectly
        # Define the discrete action space (possible shares to trade in an interval)
        self.action_space = np.linspace(0, total_shares / T_horizon_intervals * 2, action_buckets)
        self.action_space = np.round(self.action_space / (total_shares / T_horizon_intervals / action_buckets)) * \
                            (total_shares / T_horizon_intervals / action_buckets)
        self.action_space = self.action_space[self.action_space >= 0]
        self.n_actions = len(self.action_space) # Number of discrete actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate_start = exploration_start
        self.exploration_rate_end = exploration_end
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip

        self.input_dim = 4 # State: [normalized_time, normalized_inventory, normalized_price, normalized_volatility]
        self.policy_net = DQNAgent(self.input_dim, self.n_actions) # Q-network to learn policy
        self.target_net = DQNAgent(self.input_dim, self.n_actions) # Target network for stability
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Set target network to evaluation mode (no gradient updates)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(replay_buffer_size) # Experience replay buffer

        self.episode_num = 0
        self.policy_schedule = np.zeros(T_horizon_intervals) # Stores the derived policy after training

        self.market_price_range = None
        self.market_vol_range = None

    def set_market_ranges(self, price_min, price_max, vol_min, vol_max):
        """Sets the ranges for state normalization."""
        self.market_price_range = (price_min, price_max)
        self.market_vol_range = (vol_min, vol_max)

    def _normalize_state(self, time_step, current_price, current_inventory, current_vol):
        """Normalizes the continuous state variables to a [0, 1] range."""
        norm_time = time_step / (self.T_horizon_intervals - 1) # Normalize time step
        norm_inv = current_inventory / self.total_shares # Normalize inventory


        if self.market_price_range is None or self.market_vol_range is None:
            logger.warning("Market ranges not set for RL state normalization. Using rough scale.")
            norm_price = current_price / (TOTAL_SHARES_TO_TRADE * 2) # Fallback rough scaling
            norm_vol = current_vol / (HESTON_VOL_INITIAL * 2)
        else:
            min_p, max_p = self.market_price_range
            norm_price = (current_price - min_p) / (max_p - min_p + 1e-9)
            norm_price = np.clip(norm_price, 0, 1)

            min_v, max_v = self.market_vol_range
            norm_vol = (current_vol - min_v) / (max_v - min_v + 1e-9)
            norm_vol = np.clip(norm_vol, 0, 1)

        return torch.tensor([norm_time, norm_inv, norm_price, norm_vol], dtype=torch.float32)

    def _choose_action(self, state_tensor, current_inventory):
        """
        Chooses an action using an epsilon-greedy policy.
        Epsilon decays over time to balance exploration and exploitation.
        """
        # Calculate current epsilon for exploration
        eps = self.exploration_rate_end + (self.exploration_rate_start - self.exploration_rate_end) * \
              np.exp(-self.exploration_decay * self.episode_num)

        if random.random() < eps:
            # Explore: choose a random action
            action_idx = random.randrange(self.n_actions)
        else:
            # Exploit: choose the action with the highest Q-value from the policy network
            with torch.no_grad(): # No gradient calculation needed for action selection
                action_idx = self.policy_net(state_tensor.unsqueeze(0)).argmax().item()

        shares_to_trade = self.action_space[action_idx]
        # Ensure action is valid (don't trade more than available, and non-negative)
        shares_to_trade = min(shares_to_trade, current_inventory)
        shares_to_trade = max(0.0, shares_to_trade)

        return action_idx, shares_to_trade

    def _calculate_reward(self, mid_price, shares_traded, effective_execution_price,
                          is_terminal_state=False, remaining_inventory=0.0):
        """
        Calculates the immediate reward for the current action.
        Reward is based on the negative of the cost incurred in this step,
        including trade costs and opportunity cost for remaining inventory.
        """
        # Cost from the trade itself (spread + temporary impact)
        # effective_execution_price is already adjusted for spread and temporary impact
        trade_cost = (mid_price - effective_execution_price) * shares_traded

        # Opportunity cost for holding remaining inventory for this period
        # This cost is incurred for the inventory carried *into* the next period (or liquidated at the end of this)
        opportunity_cost = self.opportunity_cost_per_day_factor * remaining_inventory * mid_price

        total_cost_this_step = trade_cost + opportunity_cost

        # Scale the total cost before converting to reward
        reward = -total_cost_this_step / self.reward_scale_base

        if is_terminal_state and remaining_inventory > 1e-9:
            # Using the adjusted terminal penalty factor, also scaled
            reward -= (TERMINAL_INVENTORY_PENALTY_FACTOR * remaining_inventory * mid_price) / self.reward_scale_base

        return reward

    def _optimize_model(self):
        """
        Performs one optimization step on the DQN.
        Samples from replay buffer, computes Q-values, and updates policy network.
        """
        sample = self.memory.sample(self.batch_size)
        if sample is None: # Not enough samples in buffer yet
            return

        states, actions, rewards, next_states, dones = sample

        # Compute Q-values for current states using the policy network
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using the target network (for stability)
        with torch.no_grad(): # Don't compute gradients for target network
            next_q_values = self.target_net(next_states).max(1)[0] # Max Q-value for next state
            target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones) # 1-dones handles terminal states

        # Compute MSE loss between current and target Q-values
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Backpropagation
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip) # Clip gradients to prevent exploding gradients
        self.optimizer.step() # Update network weights

    def train_agent(self, simulation_runner, num_episodes):
        """
        Trains the DQN agent over multiple episodes.
        Each episode simulates a trading day using historical data segments.
        """
        logger.info(f"Training DRL (DQN) agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            self.episode_num = episode # Keep track of current episode for exploration decay

            # Get a random segment of historical data for the current episode
            historical_data_segment = simulation_runner._get_random_historical_segment()
            if historical_data_segment is None or len(historical_data_segment) < self.T_horizon_intervals + 1:
                logger.warning(f"Skipping episode {episode} due to insufficient historical data segment.")
                continue

            # Initialize market simulation environment for the episode
            # S0_initial and sigma_initial for the simulation environment come from the *start* of the random historical segment
            initial_sim_price = historical_data_segment.iloc[0][simulation_runner.price_col_name].item()
            initial_sim_vol_annualized = historical_data_segment.iloc[0]['DailyVolatility'].item() # Annualized vol from data

            sim_env = MarketSimulation(
                S0_initial=initial_sim_price,
                mu=self.mu, # This is daily drift
                sigma_initial=initial_sim_vol_annualized, # Annualized std dev
                dt=1.0,
                gamma_permanent=self.gamma_permanent, beta_temporary=self.beta_temporary,
                kappa_heston=self.kappa_heston, theta_heston=self.theta_heston,
                xi_heston=self.xi_heston, rho_heston=self.rho_heston,
                bid_ask_spread_bps=self.bid_ask_spread_bps,
                spread_volatility_factor=self.spread_volatility_factor
            )
            # Reset the internal state of the simulation environment with variance
            sim_env.reset(initial_price=initial_sim_price, initial_vol=initial_sim_vol_annualized**2)

            current_inventory = self.total_shares

            # The agent's observation (state) at each step comes from the market simulation's current mid-price and vol.
            # Initialize with the same starting values as sim_env.
            simulated_current_mid_price = sim_env.current_price
            simulated_current_vol = sim_env.current_vol

            total_episode_cost = 0.0 # This will track cumulative internal cost for logging

            # Simulate interaction for each time step in the horizon
            for k in range(self.T_horizon_intervals):
                is_terminal = (k == self.T_horizon_intervals - 1) # Check if it's the last step

                # State for the agent includes simulated current mid-price and vol
                state = self._normalize_state(k, simulated_current_mid_price, current_inventory, simulated_current_vol)
                action_idx, shares_to_trade_proposed = self._choose_action(state, current_inventory) # Choose action

                # Apply market constraints to the proposed trade
                shares_to_trade_actual = shares_to_trade_proposed # Start with proposed, then adjust

                # Enforce minimum trade size
                if shares_to_trade_actual > 0 and shares_to_trade_actual < MIN_SHARES_PER_SLICE:
                    if current_inventory >= MIN_SHARES_PER_SLICE:
                        shares_to_trade_actual = MIN_SHARES_PER_SLICE
                    else: # If not enough for min, trade all remaining if positive, else zero
                        shares_to_trade_actual = max(0.0, current_inventory)

                # Enforce maximum trade size
                max_shares_allowed_in_slice = TOTAL_SHARES_TO_TRADE * MAX_SHARES_PER_SLICE_FACTOR
                if shares_to_trade_actual > max_shares_allowed_in_slice:
                    shares_to_trade_actual = max_shares_allowed_in_slice

                # Final clip to remaining inventory
                shares_to_trade_actual = min(shares_to_trade_actual, current_inventory)
                shares_to_trade_actual = max(0.0, shares_to_trade_actual) # Ensure non-negative

                # Pass shares to trade to the market simulation environment.
                # The market environment handles its own stochastic evolution for next price/vol.
                current_sim_mid_price_after_step, effective_execution_price, permanent_impact_value, current_sim_vol_after_step = sim_env.step(shares_to_trade_actual, is_buy=False)

                # Reward calculation for RL agent based on the actual trade cost (mid_price - effective_exec_price)
                # and the opportunity cost
                reward = self._calculate_reward(simulated_current_mid_price, shares_to_trade_actual,
                                                effective_execution_price, is_terminal,
                                                current_inventory - shares_to_trade_actual)
                total_episode_cost -= reward # Accumulate cost (since reward is negative cost)

                current_inventory -= shares_to_trade_actual # Update inventory

                # Update the simulated current mid-price and vol for the *next* state based on the environment's step output.
                simulated_current_mid_price = current_sim_mid_price_after_step
                simulated_current_vol = current_sim_vol_after_step

                # The next_state observed by the agent uses the *newly evolved* simulated market conditions.
                next_state = self._normalize_state(k + 1, simulated_current_mid_price, current_inventory, simulated_current_vol)

                done = is_terminal or (current_inventory < 1e-9)

                # Store transition in replay buffer
                self.memory.push(state, action_idx, reward, next_state, done)
                self._optimize_model() # Perform optimization step

            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"DQN Episode {episode}/{num_episodes}, Exploration: {self.exploration_rate_end + (self.exploration_rate_start - self.exploration_rate_end) * np.exp(-self.exploration_decay * episode):.3f}, Total Cost: {total_episode_cost:.2f}")

        logger.info("DRL (DQN) training complete. Deriving optimal policy...")
        self._derive_policy_from_q_table(simulation_runner.historical_data)

    def _derive_policy_from_q_table(self, full_historical_data):
        """
        After training, derive the fixed trading schedule by greedily selecting
        actions from the learned Q-values using average market conditions.
        """
        self.policy_schedule = np.zeros(self.T_horizon_intervals)
        current_inventory = self.total_shares

        # Use average price and volatility from full historical data for policy derivation
        actual_price_column = full_historical_data.columns[0]
        avg_price = full_historical_data[actual_price_column].mean()
        avg_vol = full_historical_data['DailyVolatility'].mean()**2

        for k in range(self.T_horizon_intervals):
            if current_inventory < 1e-9:
                self.policy_schedule[k:] = 0.0
                break

            # Get state using average conditions and normalized time/inventory
            state = self._normalize_state(k, avg_price, current_inventory, avg_vol)
            with torch.no_grad(): # Greedily select action (no exploration)
                logits = self.policy_net(state.unsqueeze(0))
                action_idx = logits.argmax().item() # Use argmax for deterministic policy after training

            shares_to_trade_proposed = self.action_space[action_idx]

            # Apply market constraints to the proposed trade for final schedule derivation
            shares_to_trade_actual = shares_to_trade_proposed

            if shares_to_trade_actual > 0 and shares_to_trade_actual < MIN_SHARES_PER_SLICE:
                if current_inventory >= MIN_SHARES_PER_SLICE:
                    shares_to_trade_actual = MIN_SHARES_PER_SLICE
                else:
                    shares_to_trade_actual = max(0.0, current_inventory)

            max_shares_allowed_in_slice = TOTAL_SHARES_TO_TRADE * MAX_SHARES_PER_SLICE_FACTOR
            if shares_to_trade_actual > max_shares_allowed_in_slice:
                shares_to_trade_actual = max_shares_allowed_in_slice

            shares_to_trade_actual = min(shares_to_trade_actual, current_inventory)
            shares_to_trade_actual = max(0.0, shares_to_trade_actual)

            self.policy_schedule[k] = shares_to_trade_actual
            current_inventory -= shares_to_trade_actual

        # Re-normalize the derived schedule to ensure total shares are traded (if not already handled by clipping/rounding to zero)
        # This re-normalization might counteract the discrete nature of the new constraints, but ensures total shares are eventually sold.
        # For more strict adherence to constraints, the agent might need to learn how to adjust remaining shares.
        total_scheduled = np.sum(self.policy_schedule)
        if total_scheduled > 1e-9: # Avoid division by zero
            self.policy_schedule = (self.policy_schedule / total_scheduled) * self.total_shares
        self.policy_schedule[self.policy_schedule < 1e-9] = 0.0 # Clean up very small numbers

    def get_trading_schedule(self, *args, **kwargs):
        """Returns the derived trading schedule for the DQN agent."""
        if np.sum(self.policy_schedule) == 0 and self.total_shares > 0:
            logger.warning("DQN policy not effectively derived, defaulting to TWAP for simulation.")
            return TWAPStrategy(self.total_shares, self.T_horizon_intervals).get_trading_schedule()
        return self.policy_schedule

# --- A2C DRL AGENT IMPLEMENTATION ---

class ActorNetwork(nn.Module):
    """
    Actor network for A2C. Takes state as input and outputs logits
    (unnormalized log probabilities) for each action.
    """
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim) # Output logits for action distribution

    def forward(self, state):
        """Forward pass through the Actor network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CriticNetwork(nn.Module):
    """
    Critic network for A2C. Takes state as input and outputs a single value
    representing the estimated value of that state.
    """
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1) # Output a single value estimate

    def forward(self, state):
        """Forward pass through the Critic network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class A2CAgent(OptimalExecutionStrategy):
    """
    Implements an Actor-Critic (A2C) based optimal execution strategy.
    Learns a policy and value function concurrently to guide trading decisions.
    """
    def __init__(self, total_shares, T_horizon_intervals,
                 gamma_permanent, beta_temporary,
                 mu, kappa_heston, theta_heston, xi_heston, rho_heston,
                 action_buckets,
                 actor_lr, critic_lr, discount_factor, entropy_beta,
                 bid_ask_spread_bps, spread_volatility_factor,
                 opportunity_cost_per_day_factor,
                 reward_scale_base): # Added reward_scale_base
        super().__init__(total_shares, T_horizon_intervals)
        self.gamma_permanent = gamma_permanent
        self.beta_temporary = beta_temporary
        self.mu = mu # This mu is now expected to be daily drift
        self.kappa_heston = kappa_heston
        self.theta_heston = theta_heston
        self.xi_heston = xi_heston
        self.rho_heston = rho_heston
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.spread_volatility_factor = spread_volatility_factor
        self.opportunity_cost_per_day_factor = opportunity_cost_per_day_factor
        self.reward_scale_base = reward_scale_base # Store reward scaling factor


        # Define the discrete action space
        self.action_space = np.linspace(0, total_shares / T_horizon_intervals * 2, action_buckets)
        self.action_space = np.round(self.action_space / (total_shares / T_horizon_intervals / action_buckets)) * \
                            (total_shares / T_horizon_intervals / action_buckets)
        self.action_space = self.action_space[self.action_space >= 0]
        self.n_actions = len(self.action_space)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.entropy_beta = entropy_beta # Entropy regularization coefficient

        self.input_dim = 4 # State: [normalized_time, normalized_inventory, normalized_price, normalized_volatility]
        self.actor_net = ActorNetwork(self.input_dim, self.n_actions)
        self.critic_net = CriticNetwork(self.input_dim)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)

        self.episode_num = 0
        self.policy_schedule = np.zeros(T_horizon_intervals)

        self.market_price_range = None
        self.market_vol_range = None

    def set_market_ranges(self, price_min, price_max, vol_min, vol_max):
        """Sets the ranges for state normalization, shared with other RL agents."""
        self.market_price_range = (price_min, price_max)
        self.market_vol_range = (vol_min, vol_max)

    def _normalize_state(self, time_step, current_price, current_inventory, current_vol):
        """Normalizes the continuous state variables to a [0, 1] range."""
        norm_time = time_step / (self.T_horizon_intervals - 1)
        norm_inv = current_inventory / self.total_shares

        if self.market_price_range is None or self.market_vol_range is None:
            logger.warning("Market ranges not set for RL state normalization. Using rough scale.")
            norm_price = current_price / (TOTAL_SHARES_TO_TRADE * 2)
            norm_vol = current_vol / (HESTON_VOL_INITIAL * 2)
        else:
            min_p, max_p = self.market_price_range
            norm_price = (current_price - min_p) / (max_p - min_p + 1e-9)
            norm_price = np.clip(norm_price, 0, 1)

            min_v, max_v = self.market_vol_range
            norm_vol = (current_vol - min_v) / (max_v - min_v + 1e-9)
            norm_vol = np.clip(norm_vol, 0, 1)

        return torch.tensor([norm_time, norm_inv, norm_price, norm_vol], dtype=torch.float32)

    def _choose_action(self, state_tensor, current_inventory):
        """
        Chooses an action based on the Actor network's probability distribution.
        Samples from the categorical distribution.
        """
        with torch.no_grad(): # Action selection does not need gradients
            logits = self.actor_net(state_tensor.unsqueeze(0)) # Get logits from actor
            dist = Categorical(logits=logits) # Create a categorical distribution
            action_idx = dist.sample().item() # Sample an action

        shares_to_trade = self.action_space[action_idx]
        # Ensure action is valid (non-negative and not exceeding inventory)
        shares_to_trade = min(shares_to_trade, current_inventory)
        shares_to_trade = max(0.0, shares_to_trade)

        return action_idx, shares_to_trade

    def _calculate_reward(self, mid_price, shares_traded, effective_execution_price,
                          is_terminal_state=False, remaining_inventory=0.0):
        """
        Calculates the immediate reward for the current action for A2C.
        Reward is based on the negative of the cost incurred in this step,
        including trade costs and opportunity cost for remaining inventory.
        """
        # Cost from the trade itself (spread + temporary impact)
        trade_cost = (mid_price - effective_execution_price) * shares_traded

        # Opportunity cost for holding remaining inventory for this period
        opportunity_cost = self.opportunity_cost_per_day_factor * remaining_inventory * mid_price

        total_cost_this_step = trade_cost + opportunity_cost

        # Scale the total cost before converting to reward
        reward = -total_cost_this_step / self.reward_scale_base

        if is_terminal_state and remaining_inventory > 1e-9:
            # Using the adjusted terminal penalty factor, also scaled
            reward -= (TERMINAL_INVENTORY_PENALTY_FACTOR * remaining_inventory * mid_price) / self.reward_scale_base

        return reward

    def _learn(self, states, actions, rewards, next_states, dones):
        """
        Performs one learning step for the A2C agent.
        Calculates advantages, actor loss, and critic loss, then updates both networks.
        """
        # Convert data to tensors if not already
        states = torch.stack(states).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack(next_states).float()
        dones = torch.tensor(dones).float()

        # --- Critic Update ---
        # Get current state values from critic
        state_values = self.critic_net(states).squeeze(1)

        # Get next state values from critic (detach to prevent gradients flowing to actor via value estimation)
        with torch.no_grad():
            next_state_values = self.critic_net(next_states).squeeze(1)
            # Calculate target values: R_t + gamma * V(S_{t+1}) * (1 - done)
            target_values = rewards + self.discount_factor * next_state_values * (1 - dones)

        # Critic loss (MSE between predicted and target values)
        critic_loss = F.mse_loss(state_values, target_values)

        # Optimize critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        advantages = target_values - state_values.detach()


        logits = self.actor_net(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions) # Log probability of the chosen actions


        actor_loss = -(log_probs * advantages).mean() # Policy gradient loss
        entropy_loss = -dist.entropy().mean() # Entropy bonus to encourage exploration


        total_actor_loss = actor_loss + self.entropy_beta * entropy_loss


        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()

    def train_agent(self, simulation_runner, num_episodes):
        """
        Trains the A2C agent over multiple episodes.
        Each episode involves interacting with the market and collecting transitions,
        then updating the actor and critic networks.
        """
        logger.info(f"Training DRL (A2C) agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            self.episode_num = episode

            historical_data_segment = simulation_runner._get_random_historical_segment()
            if historical_data_segment is None or len(historical_data_segment) < self.T_horizon_intervals + 1:
                logger.warning(f"Skipping A2C episode {episode} due to insufficient historical data segment.")
                continue

            initial_sim_price = historical_data_segment.iloc[0][simulation_runner.price_col_name].item()
            initial_sim_vol_annualized = historical_data_segment.iloc[0]['DailyVolatility'].item()

            sim_env = MarketSimulation(
                S0_initial=initial_sim_price,
                mu=self.mu, # This is daily drift
                sigma_initial=initial_sim_vol_annualized, # Annualized std dev
                dt=1.0,
                gamma_permanent=self.gamma_permanent, beta_temporary=self.beta_temporary,
                kappa_heston=self.kappa_heston, theta_heston=self.theta_heston,
                xi_heston=self.xi_heston, rho_heston=self.rho_heston,
                bid_ask_spread_bps=self.bid_ask_spread_bps,
                spread_volatility_factor=self.spread_volatility_factor
            )
            sim_env.reset(initial_price=initial_sim_price, initial_vol=initial_sim_vol_annualized**2)

            current_inventory = self.total_shares

            simulated_current_mid_price = sim_env.current_price
            simulated_current_vol = sim_env.current_vol

            total_episode_cost = 0.0
            episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones = [], [], [], [], []

            for k in range(self.T_horizon_intervals):
                is_terminal = (k == self.T_horizon_intervals - 1)

                state = self._normalize_state(k, simulated_current_mid_price, current_inventory, simulated_current_vol)

                action_idx, shares_to_trade_proposed = self._choose_action(state, current_inventory)

                # Apply market constraints to the proposed trade
                shares_to_trade_actual = shares_to_trade_proposed

                # Enforce minimum trade size
                if shares_to_trade_actual > 0 and shares_to_trade_actual < MIN_SHARES_PER_SLICE:
                    if current_inventory >= MIN_SHARES_PER_SLICE:
                        shares_to_trade_actual = MIN_SHARES_PER_SLICE
                    else: # If not enough for min, trade all remaining if positive, else zero
                        shares_to_trade_actual = max(0.0, current_inventory)

                # Enforce maximum trade size
                max_shares_allowed_in_slice = TOTAL_SHARES_TO_TRADE * MAX_SHARES_PER_SLICE_FACTOR
                if shares_to_trade_actual > max_shares_allowed_in_slice:
                    shares_to_trade_actual = max_shares_allowed_in_slice

                # Final clip to remaining inventory
                shares_to_trade_actual = min(shares_to_trade_actual, current_inventory)
                shares_to_trade_actual = max(0.0, shares_to_trade_actual) # Ensure non-negative

                current_sim_mid_price_after_step, effective_execution_price, permanent_impact_value, current_sim_vol_after_step = sim_env.step(shares_to_trade_actual, is_buy=False)

                reward = self._calculate_reward(simulated_current_mid_price, shares_to_trade_actual,
                                                effective_execution_price, is_terminal,
                                                current_inventory - shares_to_trade_actual)
                total_episode_cost -= reward

                current_inventory -= shares_to_trade_actual

                simulated_current_mid_price = current_sim_mid_price_after_step
                simulated_current_vol = current_sim_vol_after_step

                next_state = self._normalize_state(k + 1, simulated_current_mid_price, current_inventory, simulated_current_vol)

                done = is_terminal or (current_inventory < 1e-9)

                # Store for batch learning (A2C typically collects full trajectories before learning)
                episode_states.append(state)
                episode_actions.append(action_idx)
                episode_rewards.append(reward)
                episode_next_states.append(next_state)
                episode_dones.append(done)

            # Perform learning step after each episode (or after a fixed number of steps)
            self._learn(episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones)

            if episode % 50 == 0:
                logger.info(f"A2C Episode {episode}/{num_episodes}, Total Cost: {total_episode_cost:.2f}")

        logger.info("DRL (A2C) training complete. Deriving optimal policy...")
        self._derive_policy_from_actor(simulation_runner.historical_data)

    def _derive_policy_from_actor(self, full_historical_data):
        """
        After A2C training, derive the fixed trading schedule by greedily selecting
        actions from the Actor network's output using average market conditions.
        """
        self.policy_schedule = np.zeros(self.T_horizon_intervals)
        current_inventory = self.total_shares

        actual_price_column = full_historical_data.columns[0]
        avg_price = full_historical_data[actual_price_column].mean()
        avg_vol = full_historical_data['DailyVolatility'].mean()**2

        for k in range(self.T_horizon_intervals):
            if current_inventory < 1e-9:
                self.policy_schedule[k:] = 0.0
                break

            state = self._normalize_state(k, avg_price, current_inventory, avg_vol)
            with torch.no_grad():
                logits = self.actor_net(state.unsqueeze(0))
                action_idx = Categorical(logits=logits).sample().item()

            shares_to_trade_proposed = self.action_space[action_idx]

            # Apply market constraints to the proposed trade for final schedule derivation
            shares_to_trade_actual = shares_to_trade_proposed

            if shares_to_trade_actual > 0 and shares_to_trade_actual < MIN_SHARES_PER_SLICE:
                if current_inventory >= MIN_SHARES_PER_SLICE:
                    shares_to_trade_actual = MIN_SHARES_PER_SLICE
                else:
                    shares_to_trade_actual = max(0.0, current_inventory)

            max_shares_allowed_in_slice = TOTAL_SHARES_TO_TRADE * MAX_SHARES_PER_SLICE_FACTOR
            if shares_to_trade_actual > max_shares_allowed_in_slice:
                shares_to_trade_actual = max_shares_allowed_in_slice

            shares_to_trade_actual = min(shares_to_trade_actual, current_inventory)
            shares_to_trade_actual = max(0.0, shares_to_trade_actual)

            self.policy_schedule[k] = shares_to_trade_actual
            current_inventory -= shares_to_trade_actual

        # Re-normalize the derived schedule to ensure total shares are traded (if not already handled by clipping/rounding to zero)


        total_scheduled = np.sum(self.policy_schedule)
        if total_scheduled > 1e-9: # Avoid division by zero
            self.policy_schedule = (self.policy_schedule / total_scheduled) * self.total_shares
        self.policy_schedule[self.policy_schedule < 1e-9] = 0.0 # Clean up very small numbers

    def get_trading_schedule(self, *args, **kwargs):
        """Returns the derived trading schedule for the A2C agent."""
        if np.sum(self.policy_schedule) == 0 and self.total_shares > 0:
            logger.warning("A2C policy not effectively derived, defaulting to TWAP for simulation.")
            return TWAPStrategy(self.total_shares, self.T_horizon_intervals).get_trading_schedule()
        return self.policy_schedule

# --- DDPG AGENT FOR CONTINUOUS ACTION SPACE ---

class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck process for adding noise to continuous actions for exploration.
    """
    def __init__(self, mu, sigma, theta, dt, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """Generate one step of the process."""
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """Reset the internal state to x0 (defaulting to zeros if not specified)."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class DDPGActor(nn.Module):
    """
    Actor network for DDPG. Takes state as input and outputs a deterministic action.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim) # Output action value
        self.max_action = max_action # Maximum possible action value (e.g., max shares to trade)

    def forward(self, state):
        """Forward pass through the Actor network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * (torch.tanh(self.fc3(x)) + 1) / 2 # Scale to [0, max_action]


class DDPGCritic(nn.Module):
    """
    Critic network for DDPG. Takes state and action as input, outputs Q-value.
    """
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256) # Concatenate state and action
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # Output single Q-value

    def forward(self, state, action):
        """Forward pass through the Critic network."""
        x = torch.cat([state, action], dim=1) # Concatenate state and action tensors
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent(OptimalExecutionStrategy):
    """
    Implements a Deep Deterministic Policy Gradient (DDPG) based optimal execution strategy
    for continuous action spaces.
    """
    def __init__(self, total_shares, T_horizon_intervals,
                 gamma_permanent, beta_temporary,
                 mu, kappa_heston, theta_heston, xi_heston, rho_heston,
                 actor_lr, critic_lr, discount_factor, tau,
                 batch_size, replay_buffer_size, gradient_clip,
                 ou_mu, ou_sigma, ou_theta, ou_dt,
                 bid_ask_spread_bps, spread_volatility_factor,
                 opportunity_cost_per_day_factor,
                 reward_scale_base): # Added reward_scale_base
        super().__init__(total_shares, T_horizon_intervals)
        self.gamma_permanent = gamma_permanent
        self.beta_temporary = beta_temporary
        self.mu = mu
        self.kappa_heston = kappa_heston
        self.theta_heston = theta_heston
        self.xi_heston = xi_heston
        self.rho_heston = rho_heston
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.spread_volatility_factor = spread_volatility_factor
        self.opportunity_cost_per_day_factor = opportunity_cost_per_day_factor
        self.reward_scale_base = reward_scale_base # Store reward scaling factor

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.tau = tau # Soft update for target networks
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip

        self.input_dim = 4 # State: [normalized_time, normalized_inventory, normalized_price, normalized_volatility]
        # Action space is 1-dimensional (shares to trade)
        self.action_dim = 1
        # Max shares an agent can propose in one interval, to be used in Actor's output scaling.
        # This is the theoretical maximum, before MIN/MAX_SHARES_PER_SLICE and current_inventory constraints.
        # Set max_action more robustly: total shares OR max shares per slice from constants, whichever is smaller
        self.max_trade_per_interval = min(TOTAL_SHARES_TO_TRADE, TOTAL_SHARES_TO_TRADE * MAX_SHARES_PER_SLICE_FACTOR * 2) # Allow for some flexibility

        self.actor = DDPGActor(self.input_dim, self.action_dim, self.max_trade_per_interval)
        self.actor_target = DDPGActor(self.input_dim, self.action_dim, self.max_trade_per_interval)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = DDPGCritic(self.input_dim, self.action_dim)
        self.critic_target = DDPGCritic(self.input_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.memory = ReplayBuffer(replay_buffer_size)

        # Ornstein-Uhlenbeck noise for exploration
        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.action_dim),
            sigma=float(ou_sigma) * np.ones(self.action_dim),
            theta=ou_theta,
            dt=ou_dt
        )
        self.episode_num = 0
        self.policy_schedule = np.zeros(T_horizon_intervals)

        self.market_price_range = None
        self.market_vol_range = None

    def set_market_ranges(self, price_min, price_max, vol_min, vol_max):
        """Sets the ranges for state normalization."""
        self.market_price_range = (price_min, price_max)
        self.market_vol_range = (vol_min, vol_max)

    def _normalize_state(self, time_step, current_price, current_inventory, current_vol):
        """Normalizes the continuous state variables to a [0, 1] range."""
        norm_time = time_step / (self.T_horizon_intervals - 1)
        norm_inv = current_inventory / self.total_shares

        if self.market_price_range is None or self.market_vol_range is None:
            logger.warning("Market ranges not set for DDPG state normalization. Using rough scale.")
            norm_price = current_price / (TOTAL_SHARES_TO_TRADE * 2) # Fallback rough scaling
            norm_vol = current_vol / (HESTON_VOL_INITIAL * 2)
        else:
            min_p, max_p = self.market_price_range
            norm_price = (current_price - min_p) / (max_p - min_p + 1e-9)
            norm_price = np.clip(norm_price, 0, 1)

            min_v, max_v = self.market_vol_range
            norm_vol = (current_vol - min_v) / (max_v - min_v + 1e-9)
            norm_vol = np.clip(norm_vol, 0, 1)

        return torch.tensor([norm_time, norm_inv, norm_price, norm_vol], dtype=torch.float32)

    def _choose_action(self, state_tensor, current_inventory, is_training):
        """
        Chooses an action based on the Actor network's output, with added noise for exploration during training.
        """
        self.actor.eval() # Set actor to evaluation mode for action selection
        with torch.no_grad():
            action = self.actor(state_tensor.unsqueeze(0)).squeeze(0).numpy() # Get deterministic action
        self.actor.train() # Set actor back to train mode

        if is_training:
            action = action + self.noise() # Add OU noise for exploration

        shares_to_trade = action.item() if self.action_dim == 1 else action[0]

        # Apply inherent action space limits and inventory limits
        shares_to_trade = np.clip(shares_to_trade, 0.0, self.max_trade_per_interval)
        shares_to_trade = min(shares_to_trade, current_inventory)
        shares_to_trade = max(0.0, shares_to_trade) # Ensure non-negative

        return torch.tensor([shares_to_trade], dtype=torch.float32) # Return as a tensor (even if scalar)

    def _calculate_reward(self, mid_price, shares_traded, effective_execution_price,
                          is_terminal_state=False, remaining_inventory=0.0):
        """
        Calculates the immediate reward for the current action.
        Reward is based on the negative of the cost incurred in this step,
        including trade costs and opportunity cost for remaining inventory.
        """
        # Cost from the trade itself (spread + temporary impact)
        trade_cost = (mid_price - effective_execution_price) * shares_traded

        # Opportunity cost for holding remaining inventory for this period
        opportunity_cost = self.opportunity_cost_per_day_factor * remaining_inventory * mid_price

        total_cost_this_step = trade_cost + opportunity_cost

        # Scale the total cost before converting to reward
        reward = -total_cost_this_step / self.reward_scale_base

        if is_terminal_state and remaining_inventory > 1e-9:
            # Using the adjusted terminal penalty factor, also scaled
            reward -= (TERMINAL_INVENTORY_PENALTY_FACTOR * remaining_inventory * mid_price) / self.reward_scale_base

        return reward

    def _learn(self):
        """
        Performs one learning step for the DDPG agent.
        Updates both Actor and Critic networks using sampled transitions.
        """
        sample = self.memory.sample(self.batch_size)
        if sample is None:
            return

        states, actions, rewards, next_states, dones = sample

        # --- Update Critic ---
        with torch.no_grad(): # Target calculations do not need gradients
            next_actions = self.actor_target(next_states) # Get actions from target actor
            target_q_values = self.critic_target(next_states, next_actions) # Q-values from target critic
            target_q_values = rewards.unsqueeze(1) + self.discount_factor * target_q_values * (1 - dones.unsqueeze(1))

        current_q_values = self.critic(states, actions) # Q-values from current critic
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Actor loss: Maximize Q-value for actions suggested by current actor (using current critic's estimate)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()

        # --- Soft Update Target Networks ---
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


    def train_agent(self, simulation_runner, num_episodes):
        """
        Trains the DDPG agent over multiple episodes.
        """
        logger.info(f"Training DRL (DDPG) agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            self.episode_num = episode
            self.noise.reset() # Reset OU noise for each episode

            historical_data_segment = simulation_runner._get_random_historical_segment()
            if historical_data_segment is None or len(historical_data_segment) < self.T_horizon_intervals + 1:
                logger.warning(f"Skipping DDPG episode {episode} due to insufficient historical data segment.")
                continue

            initial_sim_price = historical_data_segment.iloc[0][simulation_runner.price_col_name].item()
            initial_sim_vol_annualized = historical_data_segment.iloc[0]['DailyVolatility'].item()

            sim_env = MarketSimulation(
                S0_initial=initial_sim_price,
                mu=self.mu,
                sigma_initial=initial_sim_vol_annualized,
                dt=1.0,
                gamma_permanent=self.gamma_permanent, beta_temporary=self.beta_temporary,
                kappa_heston=self.kappa_heston, theta_heston=self.theta_heston,
                xi_heston=self.xi_heston, rho_heston=self.rho_heston,
                bid_ask_spread_bps=self.bid_ask_spread_bps,
                spread_volatility_factor=self.spread_volatility_factor
            )
            sim_env.reset(initial_price=initial_sim_price, initial_vol=initial_sim_vol_annualized**2)

            current_inventory = self.total_shares
            simulated_current_mid_price = sim_env.current_price
            simulated_current_vol = sim_env.current_vol

            total_episode_cost = 0.0

            for k in range(self.T_horizon_intervals):
                is_terminal = (k == self.T_horizon_intervals - 1)

                state = self._normalize_state(k, simulated_current_mid_price, current_inventory, simulated_current_vol)

                # Action comes as a tensor from _choose_action
                shares_to_trade_proposed_tensor = self._choose_action(state, current_inventory, is_training=True)
                shares_to_trade_actual = shares_to_trade_proposed_tensor.item() # Convert to scalar for simulation logic

                # Apply strict market constraints (these are hard rules, not learned by DDPG directly)
                if shares_to_trade_actual > 0 and shares_to_trade_actual < MIN_SHARES_PER_SLICE:
                    if current_inventory >= MIN_SHARES_PER_SLICE:
                        shares_to_trade_actual = MIN_SHARES_PER_SLICE
                    else:
                        shares_to_trade_actual = max(0.0, current_inventory)

                max_shares_allowed_in_slice = TOTAL_SHARES_TO_TRADE * MAX_SHARES_PER_SLICE_FACTOR
                if shares_to_trade_actual > max_shares_allowed_in_slice:
                    shares_to_trade_actual = max_shares_allowed_in_slice

                shares_to_trade_actual = min(shares_to_trade_actual, current_inventory)
                shares_to_trade_actual = max(0.0, shares_to_trade_actual)

                current_sim_mid_price_after_step, effective_execution_price, permanent_impact_value, current_sim_vol_after_step = sim_env.step(shares_to_trade_actual, is_buy=False)

                reward = self._calculate_reward(simulated_current_mid_price, shares_to_trade_actual,
                                                effective_execution_price, is_terminal,
                                                current_inventory - shares_to_trade_actual)
                total_episode_cost -= reward

                current_inventory -= shares_to_trade_actual

                simulated_current_mid_price = current_sim_mid_price_after_step
                simulated_current_vol = current_sim_vol_after_step

                next_state = self._normalize_state(k + 1, simulated_current_mid_price, current_inventory, simulated_current_vol)

                done = is_terminal or (current_inventory < 1e-9)


                self.memory.push(state, torch.tensor([shares_to_trade_actual], dtype=torch.float32), reward, next_state, done)

                # Perform learning step if enough samples are in the buffer
                if len(self.memory) > self.batch_size:
                    self._learn()

            if episode % 50 == 0:
                logger.info(f"DDPG Episode {episode}/{num_episodes}, Total Cost: {total_episode_cost:.2f}")

        logger.info("DRL (DDPG) training complete. Deriving optimal policy...")
        self._derive_policy_from_actor(simulation_runner.historical_data)

    def _derive_policy_from_actor(self, full_historical_data):
        """
        After DDPG training, derive the fixed trading schedule by greedily selecting
        actions from the Actor network's output using average market conditions.
        No noise is added here.
        """
        self.policy_schedule = np.zeros(self.T_horizon_intervals)
        current_inventory = self.total_shares

        actual_price_column = full_historical_data.columns[0]
        avg_price = full_historical_data[actual_price_column].mean()
        avg_vol = full_historical_data['DailyVolatility'].mean()**2

        for k in range(self.T_horizon_intervals):
            if current_inventory < 1e-9:
                self.policy_schedule[k:] = 0.0
                break

            state = self._normalize_state(k, avg_price, current_inventory, avg_vol)
            # Choose action without noise (pure exploitation)
            shares_to_trade_proposed_tensor = self._choose_action(state, current_inventory, is_training=False)
            shares_to_trade_actual = shares_to_trade_proposed_tensor.item()

            # Apply market constraints for final schedule
            if shares_to_trade_actual > 0 and shares_to_trade_actual < MIN_SHARES_PER_SLICE:
                if current_inventory >= MIN_SHARES_PER_SLICE:
                    shares_to_trade_actual = MIN_SHARES_PER_SLICE
                else:
                    shares_to_trade_actual = max(0.0, current_inventory)

            max_shares_allowed_in_slice = TOTAL_SHARES_TO_TRADE * MAX_SHARES_PER_SLICE_FACTOR
            if shares_to_trade_actual > max_shares_allowed_in_slice:
                shares_to_trade_actual = max_shares_allowed_in_slice

            shares_to_trade_actual = min(shares_to_trade_actual, current_inventory)
            shares_to_trade_actual = max(0.0, shares_to_trade_actual)

            self.policy_schedule[k] = shares_to_trade_actual
            current_inventory -= shares_to_trade_actual

        total_scheduled = np.sum(self.policy_schedule)
        if total_scheduled > 1e-9:
            self.policy_schedule = (self.policy_schedule / total_scheduled) * self.total_shares
        self.policy_schedule[self.policy_schedule < 1e-9] = 0.0

    def get_trading_schedule(self, *args, **kwargs):
        """Returns the derived trading schedule for the DDPG agent."""
        if np.sum(self.policy_schedule) == 0 and self.total_shares > 0:
            logger.warning("DDPG policy not effectively derived, defaulting to TWAP for simulation.")
            return TWAPStrategy(self.total_shares, self.T_horizon_intervals).get_trading_schedule()
        return self.policy_schedule


# --- MARKET SIMULATION ENVIRONMENT ---
class MarketSimulation:
    """
    Simulates the market environment including price dynamics (Heston model)
    and market impact (permanent and temporary), now incorporating a dynamic bid-ask spread.
    `current_price` now represents the mid-price.
    """
    def __init__(self, S0_initial, mu, sigma_initial, dt, gamma_permanent, beta_temporary,
                 kappa_heston, theta_heston, xi_heston, rho_heston, bid_ask_spread_bps, spread_volatility_factor):
        self.S0_initial = S0_initial # Initial mid-price for Heston simulation
        self.mu = mu # Daily drift for Heston (annualized_mu / 252)
        self.sigma_initial = sigma_initial # Annualized volatility for Heston (sqrt of initial variance)
        self.dt = dt # Time step size for SDE
        self.gamma_permanent = gamma_permanent
        self.beta_temporary = beta_temporary
        self.kappa_heston = kappa_heston
        self.theta_heston = theta_heston
        self.xi_heston = xi_heston
        self.rho_heston = rho_heston
        self.bid_ask_spread_bps = bid_ask_spread_bps # Base bid-ask spread in basis points
        self.spread_volatility_factor = spread_volatility_factor # Factor for spread's dependence on volatility

        self.current_price = S0_initial # This is the mid-price
        self.current_vol = sigma_initial**2 # Variance is volatility squared (converted from annualized std dev)

        self.price_history = [S0_initial] # Mid-price history
        self.bid_history = []
        self.ask_history = []
        self.effective_execution_price_history = [] # History of actual prices at which trades occurred
        # No longer tracking inventory/cash/costs here, as it's handled in SimulationRunner
        # to ensure consistency with overall cost calculation logic.

    def reset(self, initial_price=None, initial_vol=None):
        """Resets the simulation environment to initial conditions."""
        self.current_price = initial_price if initial_price is not None else self.S0_initial
        self.current_vol = initial_vol if initial_vol is not None else self.sigma_initial**2
        self.price_history = [self.current_price]
        self.bid_history = []
        self.ask_history = []
        self.effective_execution_price_history = []


    def step(self, shares_to_trade: float, is_buy: bool):
        """
        Simulates one time step (e.g., one day) of trading.
        Evolves mid-price and volatility stochastically using Heston SDE, then calculates
        effective execution price considering dynamic bid-ask spread and temporary impact,
        and finally applies permanent impact.
        Returns:
            new_current_mid_price (float): The mid-price after SDE evolution and permanent impact.
            effective_execution_price (float): The price at which `shares_to_trade` were executed,
                                               including dynamic bid-ask spread and temporary impact.
            permanent_impact_value (float): The magnitude of permanent impact from this trade.
            new_current_vol (float): The market volatility after SDE evolution.
        """
        # Calculate dynamic spread based on current (annualized) volatility.
        # Ensure current_vol is positive before taking sqrt. current_vol is variance.
        current_annualized_vol_std = np.sqrt(max(1e-8, self.current_vol))

        dynamic_spread_bps = self.bid_ask_spread_bps + (current_annualized_vol_std * self.spread_volatility_factor)
        dynamic_spread_bps = max(float(self.bid_ask_spread_bps), dynamic_spread_bps) # Ensure it's at least the base spread
        dynamic_spread_bps_decimal = dynamic_spread_bps / 10000.0

        half_spread = self.current_price * dynamic_spread_bps_decimal / 2.0

        # Calculate bid and ask prices (always available, defining the market)
        current_bid_price = self.current_price - half_spread
        current_ask_price = self.current_price + half_spread

        effective_execution_price = self.current_price # Default if no shares executed
        temp_impact_cost = 0.0 # Default to 0 if no trade

        if shares_to_trade > 0: # Only apply trade-specific costs if shares are actually traded
            # Base execution price considering bid/ask
            if is_buy: # Buying, execute at ask price
                effective_execution_price = self.current_price + half_spread
            else: # Selling, execute at bid price
                effective_execution_price = self.current_price - half_spread

            # Temporary market impact cost (total for the trade)
            temp_impact_cost = temporary_market_impact_quadratic_cost(shares_to_trade, self.beta_temporary)

            # Apply temporary impact per share to the effective price
            if is_buy: # Buying, temporary impact increases price
                effective_execution_price += (temp_impact_cost / shares_to_trade)
            else: # Selling, temporary impact decreases price received
                effective_execution_price -= (temp_impact_cost / shares_to_trade)

            # Ensure effective execution price remains positive
            effective_execution_price = max(effective_execution_price, 1e-8)

        # 3. Generate stochastic paths for mid-price and volatility using Heston SDE
        # SDE uses current mid-price (self.current_price) and current volatility (self.current_vol) as inputs.
        Z1_sde = np.random.randn()
        Z2_sde = np.random.randn()

        next_mid_price_before_perm_impact, next_vol_evolved = heston_sde_step(
            self.current_price, self.current_vol, self.mu, self.kappa_heston,
            self.theta_heston, self.xi_heston, self.rho_heston, self.dt, Z1_sde, Z2_sde
        )

        # 4. Apply permanent market impact to the stochastically evolved mid-price
        permanent_impact_value = permanent_market_impact_linear(shares_to_trade, self.gamma_permanent, is_buy)
        # For selling, permanent impact reduces the mid-price.
        self.current_price = next_mid_price_before_perm_impact - abs(permanent_impact_value)
        self.current_vol = next_vol_evolved

        # Ensure mid-price and volatility remain positive
        self.current_price = max(self.current_price, 1e-8)
        self.current_vol = max(self.current_vol, 1e-8)

        self.price_history.append(self.current_price) # Record updated mid-price

        return self.current_price, effective_execution_price, permanent_impact_value, self.current_vol

# --- SIMULATION RUNNER AND ANALYSIS ---
class SimulationRunner:
    """
    Manages running simulations for various optimal execution strategies
    and collects/analyzes their performance metrics.
    """
    def __init__(self, market_params, total_shares, T_horizon_intervals, num_paths, historical_data):
        self.market_params = market_params
        self.total_shares = total_shares
        self.T_horizon_intervals = T_horizon_intervals
        self.num_paths = num_paths
        self.historical_data = historical_data

        if self.T_horizon_intervals <= 0:
            raise ValueError("T_horizon_intervals must be positive.")
        # Ensure enough historical data for the simulation horizon and initial window
        if len(self.historical_data) < HISTORICAL_WINDOW_SIZE + self.T_horizon_intervals:
            logger.error("Historical data too short for simulation window and horizon.")
            raise ValueError("Insufficient historical data for simulation setup.")

        self.results = {}
        # Identify the actual price column name from the loaded historical data
        self.price_col_name = self.historical_data.columns[0]

        # Determine min/max ranges for price and volatility for normalization in RL/DP
        # Note: These are mid-price ranges.
        self.price_min = self.historical_data[self.price_col_name].min()
        self.price_max = self.historical_data[self.price_col_name].max()
        self.vol_min = self.historical_data['DailyVolatility'].min()**2
        self.vol_max = self.historical_data['DailyVolatility'].max()**2

        # Add a small buffer to min/max ranges to avoid division by zero or clipping issues
        if self.price_max - self.price_min < 1e-6:
            self.price_min = self.price_min * 0.9
            self.price_max = self.price_max * 1.1 + 1e-6
        if self.vol_max - self.vol_min < 1e-6:
            self.vol_min = self.vol_min * 0.9
            self.vol_max = self.vol_max * 1.1 + 1e-6
        self.vol_min = max(1e-8, self.vol_min) # Ensure min volatility is never zero

    def _get_random_historical_segment(self):
        """
        Extracts a random segment of historical data to be used for one simulation path.
        Ensures the segment is long enough for the trading horizon.
        """
        if len(self.historical_data) < self.T_horizon_intervals + 1:
            return None

        # Calculate maximum possible starting index for a valid segment
        max_start_idx = len(self.historical_data) - (self.T_horizon_intervals + 1)
        if max_start_idx < 0: # Not enough data for even one segment
            return None

        start_idx = random.randint(0, max_start_idx)
        end_idx = start_idx + self.T_horizon_intervals + 1 # Include one extra point for next step's base price/vol

        return self.historical_data.iloc[start_idx:end_idx].copy()


    def run_simulation(self, strategy: OptimalExecutionStrategy, strategy_name: str):
        """
        Runs Monte Carlo simulations for a given optimal execution strategy.
        Collects total costs (Implementation Shortfall) and final inventory for each simulation path.
        Also captures detailed path data for visualization for a few sample paths.
        """
        logger.info(f"\n--- Running Simulations for {strategy_name} ---")

        total_costs_all_paths = []
        final_inventory_all_paths = []
        sample_paths_data = [] # To store detailed data for visualization

        # Initial market conditions from historical data (only used if strategy is static and needs it)
        initial_overall_price = self.historical_data.iloc[0][self.price_col_name]
        initial_overall_vol = self.historical_data.iloc[0]['DailyVolatility']**2

        # Special handling for DP and A2C/DQN/DDPG to set market ranges and train if needed
        if isinstance(strategy, OptimalExecutionDynamicProgramming):
            strategy.set_market_ranges(self.price_min, self.price_max, self.vol_min, self.vol_max)
            strategy._solve_dp() # DP solves upfront
            trading_schedule_func = lambda p, v: strategy.get_trading_schedule(p, v)
        elif isinstance(strategy, (DeepReinforcementLearningAgent, A2CAgent, DDPGAgent)):
            strategy.set_market_ranges(self.price_min, self.price_max, self.vol_min, self.vol_max)
            # DRL agents need training before getting the schedule for simulation (training is called in main)
            trading_schedule_func = lambda p, v: strategy.get_trading_schedule()
        else:
            # For fixed strategies (TWAP, VWAP, Almgren-Chriss)
            trading_schedule_func = lambda p, v: strategy.get_trading_schedule()

        # For static strategies (TWAP, VWAP, Almgren-Chriss), compute schedule once
        if not (isinstance(strategy, OptimalExecutionDynamicProgramming) or
                isinstance(strategy, DeepReinforcementLearningAgent) or
                isinstance(strategy, A2CAgent) or
                isinstance(strategy, DDPGAgent)):
             # These strategies calculate their schedule once based on overall initial market conditions (or their internal rules)
             trading_schedule_static = trading_schedule_func(initial_overall_price, initial_overall_vol)
        else:
             trading_schedule_static = None # Dynamic for DP/RL, derived after training for RL

        # Initialize base market simulation environment (uses global Heston parameters for its dynamics,
         # and is reset with specific historical initial conditions for each path)
        sim_env_base = MarketSimulation(
            S0_initial=self.market_params['S0'], # This S0_initial for __init__ is a placeholder, overridden by reset
            mu=self.market_params['MU'], dt=1.0, # MU is now correctly daily
            sigma_initial=self.market_params['VOL_INITIAL'], # Placeholder, overridden by reset
            gamma_permanent=self.market_params['GAMMA_PERMANENT_IMPACT'],
            beta_temporary=self.market_params['BETA_TEMPORARY_IMPACT'],
            kappa_heston=self.market_params['H_KAPPA'], theta_heston=self.market_params['H_THETA'],
            xi_heston=self.market_params['H_XI'], rho_heston=self.market_params['H_RHO'],
            bid_ask_spread_bps=self.market_params['BID_ASK_SPREAD_BPS'],
            spread_volatility_factor=self.market_params['SPREAD_VOLATILITY_FACTOR']
        )

        # Run multiple Monte Carlo paths
        for path_idx in range(self.num_paths):
            # Get a random segment of historical data for initial conditions of this specific path
            historical_segment = self._get_random_historical_segment()
            if historical_segment is None:
                logger.warning(f"Skipping path {path_idx} for {strategy_name} due to insufficient historical data segment.")
                continue

            # Reset simulation environment for each path with initial conditions
            initial_segment_price = historical_segment.iloc[0][self.price_col_name].item()
            initial_segment_vol_annualized = historical_segment.iloc[0]['DailyVolatility'].item()
            sim_env_base.reset(initial_price=initial_segment_price,
                                initial_vol=initial_segment_vol_annualized**2)
            current_inventory = self.total_shares

            # --- Initialize cost components for this path using Implementation Shortfall definition ---
            # Initial portfolio value: Shares to trade * initial market mid-price (at start of path)
            initial_benchmark_mid_price_for_path = initial_segment_price
            initial_benchmark_value = self.total_shares * initial_benchmark_mid_price_for_path

            # Total cash received from selling shares over the horizon
            total_cash_received = 0.0

            # Initialize history for this path for visualization
            path_price_history = [initial_segment_price] # Price history from MarketSimulation (start price + T subsequent prices)
            path_inventory_history = [current_inventory]
            path_trades = []

            # Get the trading schedule for the current path
            if trading_schedule_static is not None:
                current_path_schedule = trading_schedule_static
            elif isinstance(strategy, OptimalExecutionDynamicProgramming):
                # DP schedule depends on initial mid-price/vol for that specific path
                current_path_schedule = trading_schedule_func(sim_env_base.current_price, sim_env_base.current_vol)
            elif isinstance(strategy, (DeepReinforcementLearningAgent, A2CAgent, DDPGAgent)):
                # RL schedule is derived from trained policy, and is typically static after training
                current_path_schedule = strategy.get_trading_schedule()
            else:
                # Fallback, though handled by static schedule above
                current_path_schedule = trading_schedule_func(sim_env_base.current_price, sim_env_base.current_vol)

            # Check if strategy produced a valid schedule
            if np.sum(current_path_schedule) == 0 and self.total_shares > 0:
                logger.warning(f"{strategy_name} generated a zero trading schedule for path {path_idx}. Penalizing for unexecuted shares.")
                # If no shares traded, the entire initial value is the cost (as nothing was liquidated, or assumed liquidated at 0 price)
                final_cost_for_path = initial_benchmark_value # If nothing is sold, the entire value is "lost" as shortfall
                final_inventory = self.total_shares

                # For zero schedule, fill histories for visualization
                # Price history already has initial. Extend for T_HORIZON_DAYS steps.
                path_price_history.extend([initial_segment_price] * self.T_horizon_intervals)
                path_inventory_history.extend([self.total_shares] * self.T_horizon_intervals)
                path_trades.extend([0.0] * self.T_horizon_intervals)

            else:
                # Execute trades according to the schedule for each interval
                for k in range(self.T_horizon_intervals):
                    shares_in_interval_proposed = current_path_schedule[k]
                    shares_to_execute = shares_in_interval_proposed # Start with proposed, then apply constraints

                    # Apply order sizing constraints
                    if shares_to_execute > 0:
                        # Enforce minimum trade size
                        if shares_to_execute < MIN_SHARES_PER_SLICE:
                            if current_inventory >= MIN_SHARES_PER_SLICE:
                                shares_to_execute = MIN_SHARES_PER_SLICE
                            else: # If not enough for min, trade all remaining if positive, else zero
                                shares_to_execute = max(0.0, current_inventory)

                        # Enforce maximum trade size
                        max_shares_allowed_in_slice = self.total_shares * MAX_SHARES_PER_SLICE_FACTOR
                        if shares_to_execute > max_shares_allowed_in_slice:
                            shares_to_execute = max_shares_allowed_in_slice

                    # Final clip to remaining inventory (most important constraint)
                    shares_to_execute = min(shares_to_execute, current_inventory)
                    shares_to_execute = max(0.0, shares_to_execute) # Ensure non-negative

                    # Perform a step in the stochastic market simulation environment.
                    # MarketSimulation.step now returns the *effective price of this trade*
                    current_sim_mid_price_after_step, effective_execution_price_for_trade, permanent_impact_value, current_sim_vol_after_step = \
                        sim_env_base.step(shares_to_execute, False) # is_buy=False for selling shares

                    # Accumulate cash received from this trade
                    cash_from_this_trade = shares_to_execute * effective_execution_price_for_trade
                    total_cash_received += cash_from_this_trade

                    current_inventory -= shares_to_execute # Decrease inventory

                    # Record histories for visualization
                    # path_price_history is handled by sim_env_base.price_history
                    path_inventory_history.append(current_inventory)
                    path_trades.append(shares_to_execute)

                    # Accumulate opportunity cost for remaining inventory at the end of this period
                    # This is added to the implementation shortfall calculation here in the SimulationRunner
                    opportunity_cost_this_step = self.market_params['OPPORTUNITY_COST_PER_DAY_FACTOR'] * current_inventory * current_sim_mid_price_after_step
                    total_cash_received -= opportunity_cost_this_step # Opportunity cost effectively reduces cash received


                # --- Calculate Final Cost for the path (Implementation Shortfall) ---
                # Final Portfolio Value = Cash from trades + Value of remaining inventory (at final market mid-price)
                # Implementation Shortfall = Initial Portfolio Value - Final Portfolio Value

                # Value of remaining inventory at the final simulated market mid-price
                value_of_remaining_inventory = current_inventory * sim_env_base.current_price

                final_cost_for_path = initial_benchmark_value - (total_cash_received + value_of_remaining_inventory)

                final_inventory = current_inventory # This is the actual remaining inventory

            total_costs_all_paths.append(final_cost_for_path)
            final_inventory_all_paths.append(final_inventory)

            # Store data for visualization if this is a sample path
            if path_idx < NUM_SAMPLE_VIZ_PATHS:
                sample_paths_data.append({
                    "prices": [p for p in sim_env_base.price_history], # Full price path including initial and final
                    "inventory": [inv for inv in path_inventory_history], # Full inventory path including initial and final
                    "trades": [t for t in path_trades] # Shares executed in each interval
                })

            # Log progress for stochasticity verification
            if path_idx % (self.num_paths // 10) == 0 or path_idx < 5: # Log first few paths and then periodically
                logger.info(f"  Path {path_idx}/{self.num_paths} complete. Total Cost: {final_cost_for_path:,.2f}. Remaining Inv: {final_inventory:.2f}")

        # Store results for the current strategy
        self.results[strategy_name] = {
            'costs': np.array(total_costs_all_paths),
            'final_inventory': np.array(final_inventory_all_paths),
            'strategy_object': strategy,
            'trading_schedule': [round(s, 2) for s in current_path_schedule], # Store rounded schedule
            'sample_paths': sample_paths_data # Add sampled path data for visualization
        }
        logger.info(f"Simulations for {strategy_name} finished. Mean Total Cost: {np.mean(total_costs_all_paths):,.2f}")

    def analyze_results(self):
        """Prints a summary and comparison of simulation results for all strategies."""
        logger.info("\n--- Simulation Results Summary ---")
        for name, data in self.results.items():
            costs = data['costs']
            final_inv = data['final_inventory']
            schedule = data['trading_schedule']

            logger.info(f"\nStrategy: {name}")
            logger.info(f"  Description: {data['strategy_object']}")
            # Print truncated schedule for readability if long
            if len(schedule) > 10:
                logger.info(f"  Trading Schedule (first 5, last 5 shares): {schedule[:5]} ... {schedule[-5:]}")
            else:
                logger.info(f"  Trading Schedule: {schedule}")
            logger.info(f"  Mean Total Execution Cost: {np.mean(costs):,.2f}")
            logger.info(f"  Std Dev of Total Execution Cost: {np.std(costs):,.2f}")
            logger.info(f"  Min Total Execution Cost: {np.min(costs):,.2f}")
            logger.info(f"  Max Total Execution Cost: {np.max(costs):,.2f}")
            logger.info(f"  Mean Final Remaining Inventory: {np.mean(final_inv):.4f}")
            logger.info(f"  Paths with Non-Zero Final Inventory: {np.sum(final_inv > 1e-9)} / {self.num_paths}")

        logger.info("\n--- Strategy Comparison ---")
        logger.info(f"{'Strategy':<30} | {'Mean Cost':>15} | {'Cost Std Dev':>15} | {'Avg Final Inv':>15}")
        logger.info("-" * 80)
        # Sort strategies by mean cost for easy comparison
        sorted_strategies = sorted(self.results.items(), key=lambda item: np.mean(item[1]['costs']))
        for name, data in sorted_strategies:
            logger.info(f"{name:<30} | {np.mean(data['costs']):>15,.2f} | {np.std(data['costs']):>15,.2f} | {np.mean(data['final_inventory']):>15,.4f}")
        logger.info("-" * 80)

# --- PLOTTING FUNCTIONS ---
def plot_trading_schedules(results, T_horizon_intervals):
    """Plots the trading schedules for each strategy as bar charts."""
    num_strategies = len(results)
    cols = 3 # Number of columns for subplots
    rows = math.ceil(num_strategies / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), constrained_layout=True)
    axes = axes.flatten() # Flatten for easy iteration

    x_labels = [f"Day {i+1}" for i in range(T_horizon_intervals)]

    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        schedule = np.array(data['trading_schedule']) # Ensure it's a numpy array

        ax.bar(x_labels, schedule, color='skyblue')
        ax.set_title(f"{name} Schedule", fontsize=12)
        ax.set_xlabel("Time Interval (Day)", fontsize=10)
        ax.set_ylabel("Shares", fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.yaxis.get_major_formatter().set_scientific(False) # Prevent scientific notation
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d')) # Display as integers

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Optimal Execution Trading Schedules", fontsize=16, y=1.02)
    plt.show()

def plot_inventory_paths(results, total_shares, T_horizon_intervals, num_sample_paths):
    """Plots inventory paths over time for sample paths from each strategy."""
    fig, ax = plt.subplots(figsize=(12, 7))

    time_points = np.arange(T_horizon_intervals + 1) # Including initial (Day 0) and final (Day T)

    # Use plt.colormaps.get_cmap directly
    colors_cmap = plt.colormaps.get_cmap('tab10')

    color_idx_offset = 0 # Offset for overall strategies in case some are skipped

    for name, data in results.items():
        # Adjust color index to ensure distinct colors across all strategies
        # This assumes the order of strategies is consistent
        num_strategies_in_results = len(results)

        for i in range(min(num_sample_paths, len(data['sample_paths']))):
            path_data = data['sample_paths'][i]

            # Ensure inventory history has T+1 points
            if len(path_data['inventory']) == T_horizon_intervals + 1:
                # Calculate a unique color index for each path across all strategies
                effective_color_index = (list(results.keys()).index(name) * num_sample_paths + i)
                color_normalized = effective_color_index / (num_strategies_in_results * num_sample_paths)

                ax.plot(time_points, path_data['inventory'], label=f"{name} - Path {i+1}",
                        color=colors_cmap(color_normalized), # Get color from colormap
                        linewidth=1.5, marker='o', markersize=3)
            else:
                logger.warning(f"Inventory history for {name} path {i+1} has {len(path_data['inventory'])} points, expected {T_horizon_intervals + 1}. Skipping.")


    ax.set_title("Inventory Paths Over Time (Sampled Paths)", fontsize=14)
    ax.set_xlabel("Time (Day)", fontsize=12)
    ax.set_ylabel("Remaining Inventory (Shares)", fontsize=12)
    ax.set_xticks(time_points)
    ax.set_ylim(bottom=0, top=total_shares * 1.05) # Ensure y-axis starts at 0 and goes slightly above total shares
    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.2, 1)) # Place legend outside
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_price_paths_with_trades(results, T_horizon_intervals, num_sample_paths):
    """Plots mid-price paths with trade markers for sample paths from each strategy."""
    fig, ax = plt.subplots(figsize=(12, 7))

    time_points = np.arange(T_horizon_intervals + 1) # Including initial (Day 0) and final (Day T)
    trade_x_points = np.arange(1, T_horizon_intervals + 1) # Trades occur after Day 0, at Day 1, Day 2, etc.

    colors_cmap = plt.colormaps.get_cmap('tab10')

    # Determine global max trade volume for consistent marker scaling
    all_trade_volumes = []
    for name, data in results.items():
        for path_data in data['sample_paths']:
            all_trade_volumes.extend([t for t in path_data['trades'] if t > 0])
    global_max_trade_vol = max(all_trade_volumes) if all_trade_volumes else 1 # Avoid division by zero


    for name, data in results.items():
        # Adjust color index to ensure distinct colors across all strategies
        num_strategies_in_results = len(results)

        for i in range(min(num_sample_paths, len(data['sample_paths']))):
            path_data = data['sample_paths'][i]

            # Ensure price history has T+1 points and trades has T points
            if len(path_data['prices']) == T_horizon_intervals + 1 and len(path_data['trades']) == T_horizon_intervals:
                effective_color_index = (list(results.keys()).index(name) * num_sample_paths + i)
                color_normalized = effective_color_index / (num_strategies_in_results * num_sample_paths)

                # Plot price path
                ax.plot(time_points, path_data['prices'], label=f"{name} Price Path - Path {i+1}",
                        color=colors_cmap(color_normalized),
                        linewidth=1.5, alpha=0.7)

                # Plot trades as scatter points
                trade_prices = []
                trade_volumes_for_plot = []
                for t_idx, trade_vol in enumerate(path_data['trades']):
                    if trade_vol > 0:
                        # Trade occurs during interval k (Day k+1), use price at end of interval
                        trade_prices.append(path_data['prices'][t_idx + 1])
                        trade_volumes_for_plot.append(trade_vol)
                    else:
                        trade_prices.append(np.nan) # No trade, no point
                        trade_volumes_for_plot.append(np.nan)

                # Scale marker size by shares traded (using a linear scale for now, can be log)
                marker_sizes = [500 * (vol / global_max_trade_vol) if vol > 0 else 0 for vol in trade_volumes_for_plot]

                ax.scatter(trade_x_points, trade_prices,
                           s=marker_sizes,
                           label=f"{name} Trades - Path {i+1}",
                           color=colors_cmap(color_normalized),
                           marker='v', alpha=0.9, zorder=5) # zorder to ensure markers are on top
            else:
                logger.warning(f"Price/Trade history for {name} path {i+1} has inconsistent length. Skipping.")


    ax.set_title("Mid-Price Paths with Trades (Sampled Paths)", fontsize=14)
    ax.set_xlabel("Time (Day)", fontsize=12)
    ax.set_ylabel("Mid-Price ($)", fontsize=12)
    ax.set_xticks(time_points)
    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.2, 1))
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- MAIN EXECUTION AND DEMONSTRATION ---
if __name__ == "__main__":
    logger.info("--- Starting Optimal Execution Project ---")
    start_time_total = time.time()

    # Load historical market data
    historical_data = load_historical_data(TICKER, START_DATE, END_DATE)
    if historical_data is None:
        logger.critical("Failed to load historical data. Exiting.")
        sys.exit(1)

    # Determine the actual price column used by load_historical_data (e.g., 'Close' or 'Adj Close')
    actual_price_column = historical_data.columns[0]

    # Calculate the global reward scaling base once
    # This ensures rewards are normalized consistently across all RL agents.
    if not historical_data[actual_price_column].empty:
        avg_historical_price = historical_data[actual_price_column].mean()
        # REWARD_SCALE_BASE will be the approximate notional value of the total trade
        REWARD_SCALE_BASE = TOTAL_SHARES_TO_TRADE * avg_historical_price
        if REWARD_SCALE_BASE < 1e-9: # Prevent division by zero or very small numbers
            REWARD_SCALE_BASE = 1.0 # Fallback
    else:
        REWARD_SCALE_BASE = 1.0 # Fallback if historical data is empty


    # Configure market parameters for the simulation environment
    market_config = {
        'S0': historical_data.iloc[0][actual_price_column], # Initial price for Heston simulation (placeholder for MarketSimulation __init__)
        'MU': HESTON_MU / 252.0, # CRITICAL FIX: Use daily drift for Heston model
        'VOL_INITIAL': HESTON_VOL_INITIAL, # Global Heston initial volatility (annualized)
        'H_KAPPA': HESTON_KAPPA,
        'H_THETA': HESTON_THETA,
        'H_XI': HESTON_XI,
        'H_RHO': HESTON_RHO,
        'GAMMA_PERMANENT_IMPACT': GAMMA_PERMANENT_IMPACT,
        'BETA_TEMPORARY_IMPACT': BETA_TEMPORARY_IMPACT,
        'BID_ASK_SPREAD_BPS': BID_ASK_SPREAD_BPS,
        'SPREAD_VOLATILITY_FACTOR': SPREAD_VOLATILITY_FACTOR,
        'OPPORTUNITY_COST_PER_DAY_FACTOR': OPPORTUNITY_COST_PER_DAY_FACTOR # New: Opportunity Cost
    }

    # Initialize the simulation runner
    sim_runner = SimulationRunner(
        market_params=market_config,
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS,
        num_paths=NUM_MONTE_CARLO_PATHS,
        historical_data=historical_data
    )

    # --- Run TWAP Strategy ---
    twap_strategy = TWAPStrategy(
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS
    )
    sim_runner.run_simulation(twap_strategy, "TWAP")

    # --- Run VWAP Strategy ---
    vwap_strategy = VWAPStrategy(
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS,
        volume_profile_type='linear_peak_middle'
    )
    sim_runner.run_simulation(vwap_strategy, "VWAP (Linear Peak)")

    # --- Run Almgren-Chriss Strategy ---
    # Adjusted RISK_AVERSION_PARAMETER to encourage trading and demonstrate its behavior
    RISK_AVERSION_PARAMETER = 1e-3
    if 'DailyVolatility' in historical_data.columns and not historical_data['DailyVolatility'].empty:
        # Using mean daily volatility derived from historical data for AC sigma_daily
        avg_daily_vol_for_ac = historical_data['DailyVolatility'].mean() / np.sqrt(252) # Convert to daily std dev
    else:
        # Fallback if historical volatility is not available
        avg_daily_vol_for_ac = HESTON_VOL_INITIAL / np.sqrt(252)

    ac_strategy = AlmgrenChrissStrategy(
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS,
        sigma_daily=avg_daily_vol_for_ac,
        risk_aversion=RISK_AVERSION_PARAMETER,
        gamma_ac=GAMMA_PERMANENT_IMPACT,
        eta_ac=BETA_TEMPORARY_IMPACT
    )
    sim_runner.run_simulation(ac_strategy, "Almgren-Chriss")

    # --- Run Dynamic Programming Strategy ---
    dp_strategy = OptimalExecutionDynamicProgramming(
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS,
        gamma_permanent=GAMMA_PERMANENT_IMPACT,
        beta_temporary=BETA_TEMPORARY_IMPACT,
        mu=market_config['MU'], # Pass the corrected daily MU
        kappa_heston=HESTON_KAPPA, theta_heston=HESTON_THETA, xi_heston=HESTON_XI, rho_heston=HESTON_RHO,
        price_buckets=DP_RL_PRICE_BUCKETS, inventory_buckets=DP_RL_INVENTORY_BUCKETS,
        action_buckets=DP_RL_ACTION_BUCKETS, vol_buckets=DP_RL_VOL_BUCKETS,
        bid_ask_spread_bps=market_config['BID_ASK_SPREAD_BPS'],
        spread_volatility_factor=market_config['SPREAD_VOLATILITY_FACTOR'],
        opportunity_cost_per_day_factor=market_config['OPPORTUNITY_COST_PER_DAY_FACTOR'] # Corrected keyword
    )
    # Set market ranges for DP's state discretization
    dp_strategy.set_market_ranges(sim_runner.price_min, sim_runner.price_max, sim_runner.vol_min, sim_runner.vol_max)
    sim_runner.run_simulation(dp_strategy, "Dynamic Programming")

    # --- Run Deep Reinforcement Learning (DQN) Strategy ---
    drl_agent = DeepReinforcementLearningAgent(
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS,
        gamma_permanent=GAMMA_PERMANENT_IMPACT,
        beta_temporary=BETA_TEMPORARY_IMPACT,
        mu=market_config['MU'], # Pass the corrected daily MU
        kappa_heston=HESTON_KAPPA, theta_heston=HESTON_THETA, xi_heston=HESTON_XI, rho_heston=HESTON_RHO,
        inventory_buckets=DP_RL_INVENTORY_BUCKETS, action_buckets=DP_RL_ACTION_BUCKETS,
        learning_rate=DQN_LEARNING_RATE, discount_factor=DQN_DISCOUNT_FACTOR,
        exploration_start=DQN_EXPLORATION_RATE_START, exploration_end=DQN_EXPLORATION_RATE_END,
        exploration_decay=DQN_EXPLORATION_DECAY, batch_size=DQN_BATCH_SIZE,
        replay_buffer_size=DQN_REPLAY_BUFFER_SIZE, target_update_freq=DQN_TARGET_UPDATE_FREQ,
        gradient_clip=DQN_GRADIENT_CLIP,
        bid_ask_spread_bps=market_config['BID_ASK_SPREAD_BPS'],
        spread_volatility_factor=market_config['SPREAD_VOLATILITY_FACTOR'],
        opportunity_cost_per_day_factor=market_config['OPPORTUNITY_COST_PER_DAY_FACTOR'],
        reward_scale_base=REWARD_SCALE_BASE # Pass the calculated reward scale base
    )
    # Set market ranges for DQN's state normalization
    drl_agent.set_market_ranges(sim_runner.price_min, sim_runner.price_max, sim_runner.vol_min, sim_runner.vol_max)
    drl_agent.train_agent(sim_runner, DQN_EPISODES) # Train the DQN agent
    sim_runner.run_simulation(drl_agent, "Deep Reinforcement Learning (DQN)")

    # --- Run Actor-Critic (A2C) Deep Reinforcement Learning Strategy ---
    a2c_agent = A2CAgent(
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS,
        gamma_permanent=GAMMA_PERMANENT_IMPACT,
        beta_temporary=BETA_TEMPORARY_IMPACT,
        mu=market_config['MU'], # Pass the corrected daily MU
        kappa_heston=HESTON_KAPPA, theta_heston=HESTON_THETA, xi_heston=HESTON_XI, rho_heston=HESTON_RHO,
        action_buckets=DP_RL_ACTION_BUCKETS, # Uses same action space discretization
        actor_lr=A2C_ACTOR_LR,
        critic_lr=A2C_CRITIC_LR,
        discount_factor=A2C_DISCOUNT_FACTOR,
        entropy_beta=A2C_ENTROPY_BETA,
        bid_ask_spread_bps=market_config['BID_ASK_SPREAD_BPS'],
        spread_volatility_factor=market_config['SPREAD_VOLATILITY_FACTOR'],
        opportunity_cost_per_day_factor=market_config['OPPORTUNITY_COST_PER_DAY_FACTOR'],
        reward_scale_base=REWARD_SCALE_BASE # Pass the calculated reward scale base
    )
    # Set market ranges for A2C's state normalization
    a2c_agent.set_market_ranges(sim_runner.price_min, sim_runner.price_max, sim_runner.vol_min, sim_runner.vol_max)
    a2c_agent.train_agent(sim_runner, A2C_EPISODES) # Train the A2C agent
    sim_runner.run_simulation(a2c_agent, "Deep Reinforcement Learning (A2C)")

    # Run Deep Deterministic Policy Gradient (DDPG) Strategy
    ddpg_agent = DDPGAgent(
        total_shares=TOTAL_SHARES_TO_TRADE,
        T_horizon_intervals=T_HORIZON_DAYS,
        gamma_permanent=GAMMA_PERMANENT_IMPACT,
        beta_temporary=BETA_TEMPORARY_IMPACT,
        mu=market_config['MU'],
        kappa_heston=HESTON_KAPPA, theta_heston=HESTON_THETA, xi_heston=HESTON_XI, rho_heston=HESTON_RHO,
        actor_lr=DDPG_ACTOR_LR,
        critic_lr=DDPG_CRITIC_LR,
        discount_factor=DDPG_DISCOUNT_FACTOR,
        tau=DDPG_TAU,
        batch_size=DDPG_BATCH_SIZE,
        replay_buffer_size=DDPG_REPLAY_BUFFER_SIZE,
        gradient_clip=DDPG_GRADIENT_CLIP,
        ou_mu=DDPG_OU_MU, ou_sigma=DDPG_OU_SIGMA, ou_theta=DDPG_OU_THETA, ou_dt=DDPG_OU_DT, # dt for OU noise is 1.0 (daily)
        bid_ask_spread_bps=market_config['BID_ASK_SPREAD_BPS'],
        spread_volatility_factor=market_config['SPREAD_VOLATILITY_FACTOR'],
        opportunity_cost_per_day_factor=market_config['OPPORTUNITY_COST_PER_DAY_FACTOR'],
        reward_scale_base=REWARD_SCALE_BASE
    )
    # Set market ranges for DDPG's state normalization
    ddpg_agent.set_market_ranges(sim_runner.price_min, sim_runner.price_max, sim_runner.vol_min, sim_runner.vol_max)
    ddpg_agent.train_agent(sim_runner, DDPG_EPISODES) # Train the DDPG agent
    sim_runner.run_simulation(ddpg_agent, "Deep Reinforcement Learning (DDPG)")


    # --- Analyze and Compare Results ---
    sim_runner.analyze_results()

    end_time_total = time.time()
    logger.info(f"\n--- Project Finished in {end_time_total - start_time_total:.2f} seconds ---")

    # --- Plotting Visualizations ---
    logger.info("\n--- Generating Visualizations ---")
    plot_trading_schedules(sim_runner.results, T_HORIZON_DAYS)
    plot_inventory_paths(sim_runner.results, TOTAL_SHARES_TO_TRADE, T_HORIZON_DAYS, NUM_SAMPLE_VIZ_PATHS)
    plot_price_paths_with_trades(sim_runner.results, T_HORIZON_DAYS, NUM_SAMPLE_VIZ_PATHS)
    logger.info("\n--- Visualizations Generated. Close windows to finish program. ---")

