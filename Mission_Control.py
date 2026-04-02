import streamlit as st
import pandas as pd
import json
import time
import re
import plotly.express as px
import alpaca_trade_api as tradeapi
import gspread
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import yfinance as yf
import psutil
import requests

st.set_page_config(
    page_title="Angel V6 Mission Control",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STYLING ===
st.markdown("""
    <style>
    /* VS Code Terminal Theme */
    .terminal-box {
        background-color: #1e1e1e; /* VS Code Background */
        color: #cccccc;            /* Default Text */
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        padding: 10px;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        height: 600px;
        overflow-y: auto;
        font-size: 14px;           /* Larger Font */
        line-height: 1.5;
    }
    .log-line {
        display: block;            /* Forces each log to its own line */
        padding: 1px 0;
        border-bottom: 1px solid #2d2d2d; /* Subtle separator line */
    }
    .log-ts { color: #6a9955; }    /* VS Code Comment Green for Dates */
    .log-info { color: #569cd6; font-weight: bold; } /* VS Code Blue */
    .log-warn { color: #cca700; font-weight: bold; } /* Yellow */
    .log-err { color: #f44747; font-weight: bold; }  /* Red */
    .log-ticker { color: #c586c0; font-weight: bold;} /* Purple for Tickers */
    </style>
""", unsafe_allow_html=True)

# === CONNECTIONS (CACHED) ===

@st.cache_resource
def init_alpaca():
    """Connects to Alpaca using Streamlit Secrets."""
    try:
        api_key = st.secrets["alpaca"]["API_KEY"]
        secret_key = st.secrets["alpaca"]["SECRET_KEY"]
        base_url = st.secrets["alpaca"]["BASE_URL"]
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        return api
    except Exception as e:
        st.error(f"Alpaca Connection Error: {e}")
        return None

# OPTIMIZATION: Cache data for 60s to prevent Google API rate limits
@st.cache_data(ttl=60)
def read_bot_logs():
    """Reads logs from Google Sheets (The Bridge)."""
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open("Angel_Bot_Logs")
        worksheet = sh.worksheet("logs")
        
        # Get all values, but filter out empty strings immediately
        logs = worksheet.col_values(1)
        clean_logs = [line for line in logs if line.strip()] 
        
        return clean_logs
    except Exception as e:
        return [f"Google Sheets Error: {e}"]

@st.cache_data(ttl=30)
def get_account_data(_api):
    try:
        # Convert Alpaca objects to simple dictionaries for safe caching
        account = _api.get_account()._raw
        positions = [p._raw for p in _api.list_positions()]
        # Increased limit to 100 to ensure we capture older entry dates for the Time-Stop feature
        orders = [o._raw for o in _api.list_orders(status='all', limit=100, direction='desc')]
        return account, positions, orders
    except:
        return None, [], []

def extract_bot_states(logs):
    """Extracts the exact number of tickers in each state from the end-of-cycle log."""
    for line in reversed(logs):
        if "Current states count" in line:
            match = re.search(r"Counter\(\{([^}]+)\}\)", line)
            if match:
                state_str = match.group(1)
                try:
                    return dict((k.strip("' "), int(v)) for k, v in (item.split(':') for item in state_str.split(',')))
                except:
                    pass
    return {}

@st.cache_data(ttl=60)
def get_portfolio_history(_api):
    try:
        # Fetch ALL history first
        history = _api.get_portfolio_history(period='all', timeframe='1D')
        
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # --- UPDATE: Filter for Start Date (24 May 2025) ---
        start_date = pd.Timestamp("2025-06-01")
        df = df[df['timestamp'] >= start_date].copy()
        
        # Sort to ensure calculations are correct
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        return pd.DataFrame()

def parse_latest_run_logic(logs):
    """
    Parses logs to extract:
    1. Signals (Decisions)
    2. Watchlist (High potential)
    3. Neural Conviction (Latest confidence score and Action)
    """
    signals = {}
    watchlist = [] 
    neural_conviction = {} 
    last_run_timestamp = None
    last_run_str = "Unknown"
    
    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})')
    conf_pattern = re.compile(r'Conf:\s*([\d\.]+)%?')
    
    ignore_tags = {'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG'}
    action_map = {"0": "HOLD", "1": "LONG", "2": "SHORT", "3": "CLOSE"}
    
    for line in reversed(logs):
        all_tags = re.findall(r'\[([A-Z]+)\]', line)
        valid_tickers = [tag for tag in all_tags if tag not in ignore_tags]
        
        if valid_tickers:
            ticker = valid_tickers[-1] 
            
            conf_match = conf_pattern.search(line)
            confidence = float(conf_match.group(1)) if conf_match else 0.0
            
            # Extract the Action State (0, 1, 2, 3)
            action_match = re.search(r'(?:PROPOSAL|SIGNAL):\s*(\d)', line)
            action_str = action_map.get(action_match.group(1), "") if action_match else ""
            
            # Store Confidence AND Action for the Chart
            if ticker not in neural_conviction and confidence > 0:
                neural_conviction[ticker] = {"Confidence": confidence, "Action": action_str}

            if ticker not in signals:
                clean_msg = line.split(f"[{ticker}]")[-1].strip()
                if "FINAL SIGNAL" in line:
                    signals[ticker] = "✅ " + clean_msg
                elif "Forcing HOLD" in line or "Margin" in line:
                    signals[ticker] = "⏸️ " + clean_msg
                    # Checks against 20.0 threshold, formats string securely
                    if confidence > 20.0: 
                        tag = "🔥 Screaming Setup" if confidence > 80.0 else ("⚡ High Conviction" if confidence > 50.0 else "👀 Watching")
                        watchlist.append({"Ticker": ticker, "Conf": f"{confidence:.1f}%", "Status": tag})
                elif "Prediction" in line:
                    signals[ticker] = "🤔 " + clean_msg
                elif "Error" in line:
                    signals[ticker] = "❌ " + clean_msg
                else:
                    if "RAW PROPOSAL" in line and confidence > 20.0:
                         tag = "🔥 Screaming Setup" if confidence > 80.0 else ("⚡ High Conviction" if confidence > 50.0 else "👀 Watching")
                         watchlist.append({"Ticker": ticker, "Conf": f"{confidence:.1f}%", "Status": tag})

        if last_run_str == "Unknown":
            match = ts_pattern.search(line)
            if match:
                last_run_str = match.group(1)
                try:
                    last_run_timestamp = datetime.strptime(last_run_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass

    unique_watchlist = {v['Ticker']:v for v in watchlist}.values()
    return last_run_str, last_run_timestamp, signals, list(unique_watchlist), neural_conviction

@st.cache_data(ttl=300)
def get_market_benchmark():
    """Fetches SPY daily return for the Alpha calculation."""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="2d")
        if len(hist) >= 2:
            return ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
        return 0.0
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_trade_excursions(_api, orders):
    """
    Parses recent filled orders to find closed 'round-trip' trades.
    Fetches historical data to calculate MAE (Max Adverse Excursion) 
    and MFE (Max Favorable Excursion) for stop-loss optimization.
    """
    if not orders: return pd.DataFrame()
    
    trades = []
    # Sort oldest to newest to build the trade timeline
    filled_orders = sorted([o for o in orders if isinstance(o, dict) and o.get('status') == 'filled'], 
                           key=lambda x: x.get('filled_at', ''))
    
    # Lightweight FIFO matcher
    inventory = {}
    for o in filled_orders:
        sym = o.get('symbol')
        side = o.get('side')
        qty = float(o.get('filled_qty', 0))
        price = float(o.get('filled_avg_price', 0))
        
        try:
            t = pd.to_datetime(o.get('filled_at')).tz_convert('UTC')
        except:
            continue
            
        if sym not in inventory:
            inventory[sym] = {'qty': 0, 'cost': 0, 'entry_time': t, 'side': None}
            
        inv = inventory[sym]
        
        # Open new position
        if inv['qty'] == 0:
            inv['side'] = side
            inv['cost'] = price
            inv['entry_time'] = t
            inv['qty'] += qty
        else:
            # Add to existing position
            if inv['side'] == side:
                inv['cost'] = ((inv['cost'] * inv['qty']) + (price * qty)) / (inv['qty'] + qty)
                inv['qty'] += qty
            # Close/Reduce position -> THIS IS A COMPLETED TRADE
            else:
                closed_qty = min(inv['qty'], qty)
                inv['qty'] -= closed_qty
                
                if closed_qty > 0:
                    trades.append({
                        'Ticker': sym,
                        'Type': 'Long' if inv['side'] == 'buy' else 'Short',
                        'Entry_Time': inv['entry_time'],
                        'Exit_Time': t,
                        'Entry_Price': inv['cost'],
                        'Exit_Price': price,
                    })
                if inv['qty'] == 0:
                    inv['side'] = None

    # Fetch highs/lows for the last 25 closed trades to avoid API limits
    recent_trades = trades[-25:]
    excursion_data = []
    
    for t in recent_trades:
        start_str = t['Entry_Time'].strftime('%Y-%m-%d')
        end_str = (t['Exit_Time'] + timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            # Suppress output so it doesn't print to terminal
            df = yf.download(t['Ticker'], start=start_str, end=end_str, progress=False)
            if not df.empty:
                # Use .values to safely extract scalar max/min regardless of yfinance multi-index formats
                trade_high = float(df['High'].values.max())
                trade_low = float(df['Low'].values.min())
                entry = t['Entry_Price']
                exit_p = t['Exit_Price']
                
                if t['Type'] == 'Long':
                    mfe = (trade_high - entry) / entry * 100
                    mae = (entry - trade_low) / entry * 100 # Keep positive for plotting scale
                    pnl = (exit_p - entry) / entry * 100
                else:
                    mfe = (entry - trade_low) / entry * 100
                    mae = (trade_high - entry) / entry * 100
                    pnl = (entry - exit_p) / entry * 100
                    
                t['MFE (%)'] = mfe
                t['MAE (%)'] = -mae # Convert to negative for the X-axis mapping
                t['PnL (%)'] = pnl
                t['Result'] = 'Win' if pnl > 0 else 'Loss'
                excursion_data.append(t)
        except Exception:
            continue
            
    return pd.DataFrame(excursion_data)

@st.cache_data(ttl=3600)
def get_correlation_matrix(tickers):
    """Generates a 30-day correlation matrix for active positions."""
    if not tickers or len(tickers) < 2: return None
    try:
        df = yf.download(tickers, period="1mo", interval="1d", progress=False)['Close']
        if isinstance(df, pd.Series): return None
        return df.corr()
    except:
        return None

def get_system_telemetry():
    """Fetches local CPU, RAM, and API latency."""
    cpu_pct = psutil.cpu_percent(interval=0.1)
    ram_pct = psutil.virtual_memory().percent
    try:
        start = time.time()
        requests.get("https://api.alpaca.markets/v2/clock", timeout=2)
        latency = int((time.time() - start) * 1000)
    except:
        latency = 999
    return cpu_pct, ram_pct, latency


def calculate_drawdown(df):
    """Calculates Drawdown % and Time Underwater (Recovery Days)."""
    df = df.copy()
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    # Calculate days spent below the high-water mark
    df['is_high'] = df['equity'] >= df['peak']
    # Groups consecutive underwater days and counts them
    df['underwater_days'] = df.groupby(df['is_high'].cumsum()).cumcount()
    
    return df

def calculate_daily_returns(df):
    """Calculates daily percentage change."""
    df = df.copy()
    df['daily_return'] = df['equity'].pct_change() * 100
    # Color logic: Green for positive, Red for negative
    df['color'] = df['daily_return'].apply(lambda x: '#00ff41' if x >= 0 else '#ff4b4b')
    return df

def calculate_seasonality(df):
    """
    Analyzes performance by Day of Week and Month of Year.
    Returns Avg Return and Win Rate for both.
    """
    s_df = df.copy()
    
    # === FIX: Normalize to US Market Time (US/Eastern) ===
    # This fixes the "Blank Monday" (AU Tuesday) and "Missing Saturday" (AU Friday)
    if s_df['timestamp'].dt.tz is None:
        # If timestamps are naive (no timezone), assume UTC first then convert
        s_df['timestamp'] = s_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:
        s_df['timestamp'] = s_df['timestamp'].dt.tz_convert('US/Eastern')
    # =====================================================

    s_df['daily_return'] = s_df['equity'].pct_change() * 100
    s_df['Day'] = s_df['timestamp'].dt.day_name()
    s_df['Month'] = s_df['timestamp'].dt.month_name()
    s_df['Month_Num'] = s_df['timestamp'].dt.month
    
    # 1. Day of Week Stats (Standard Mon-Fri Market Week)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_stats = s_df.groupby('Day')['daily_return'].agg(
        Avg_Return='mean',
        Win_Rate=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).reindex(day_order)
    
    # 2. Monthly Stats
    monthly_stats = s_df.groupby(['Month_Num', 'Month'])['daily_return'].agg(
        Avg_Return='mean',
        Win_Rate=lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index().sort_values('Month_Num').set_index('Month')
    
    return day_stats, monthly_stats

def calculate_advanced_metrics(hist_df):
    """Calculates strict Portfolio Metrics including advanced Institutional metrics."""
    if hist_df.empty: return {}
    
    df = hist_df.copy()
    df['daily_return'] = df['equity'].pct_change()
    
    # --- 1. RETURN & RISK ---
    # FIX: Added tz='UTC' to match the incoming hist_df
    start_date = pd.Timestamp("2025-05-24", tz='UTC') 
    current_date = df['timestamp'].max()
    
    # Ensure current_date is also UTC just in case
    if current_date.tz is None:
        current_date = current_date.tz_localize('UTC')
    
    days = (current_date - start_date).days
    if days < 1: days = 1
    
    total_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]
    cagr = ((1 + total_return) ** (365 / days)) - 1
    
    df['peak'] = df['equity'].cummax()
    max_dd = ((df['equity'] - df['peak']) / df['peak']).min()
    
    # MAR Ratio (Return / Risk)
    mar = (cagr / abs(max_dd)) if max_dd != 0 else 0

    # Sharpe Ratio
    mean_ret = df['daily_return'].mean()
    std_ret = df['daily_return'].std()
    sharpe = (mean_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0
    
    # Sortino Ratio
    downside_std = df[df['daily_return'] < 0]['daily_return'].std()
    sortino = (mean_ret / downside_std) * (252 ** 0.5) if downside_std > 0 else 0

    # --- 2. ADVANCED RISK (PAIN & REGIME) ---
    # We pass df to calculate_drawdown first to get the 'underwater_days' and 'drawdown' columns
    df_with_dd = calculate_drawdown(df)
    max_underwater_days = int(df_with_dd['underwater_days'].max()) if 'underwater_days' in df_with_dd.columns else 0
    
    # Ulcer Index (Quadratic Mean of Drawdowns)
    # Measures the depth and duration of drawdowns
    ulcer_index = (df_with_dd['drawdown'] ** 2).mean() ** 0.5 if 'drawdown' in df_with_dd.columns else 0

    # Information Ratio (Consistency of Alpha vs Benchmark)
    # Requires 'benchmark_return' column (e.g., SPY daily returns) to be merged into hist_df before passing
    if 'benchmark_return' in df.columns:
        active_return = df['daily_return'] - df['benchmark_return']
        tracking_error = active_return.std()
        information_ratio = (active_return.mean() / tracking_error) * (252 ** 0.5) if tracking_error > 0 else 0
    else:
        information_ratio = 0.0

    # --- 3. PROFITABILITY & QUALITY ---
    df['diff'] = df['equity'].diff()
    gross_profit = df[df['diff'] > 0]['diff'].sum()
    gross_loss = abs(df[df['diff'] < 0]['diff'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # Win Rate (Days)
    wins = len(df[df['diff'] > 0])
    total_active = len(df[df['diff'] != 0])
    win_rate = (wins / total_active) if total_active > 0 else 0
    
    # Expectancy & System Quality Number (SQN)
    avg_win = df[df['daily_return'] > 0]['daily_return'].mean()
    avg_win = avg_win if pd.notna(avg_win) else 0.0
    avg_loss = abs(df[df['daily_return'] < 0]['daily_return'].mean())
    avg_loss = avg_loss if pd.notna(avg_loss) else 0.0
    
    loss_rate = 1 - win_rate
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    
    # SQN Approximation (Using active days as N)
    sqn = (total_active ** 0.5) * (expectancy / std_ret) if std_ret > 0 else 0

    # --- OMEGA RATIO (Threshold = 0) ---
    threshold = 0.0 
    excess_returns = df['daily_return'] - threshold
    positive_sum = excess_returns[excess_returns > 0].sum()
    negative_sum = abs(excess_returns[excess_returns < 0].sum())
    
    omega_ratio = (positive_sum / negative_sum) if negative_sum > 0 else float('inf')

    return {
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Recovery Time": max_underwater_days,
        "Ulcer Index": ulcer_index,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Information Ratio": information_ratio,
        "MAR Ratio": mar,
        "Profit Factor": profit_factor,
        "Win Rate (Daily)": win_rate,
        "Expectancy": expectancy,
        "SQN": sqn,
        "Omega Ratio": omega_ratio
    }

def create_scorecard_df(metrics):
    """Formats the simplified Strategy Scorecard."""
    
    data = [
        # --- RETURN ---
        {"METRIC": "CAGR (Account)", "YOURS": f"{metrics.get('CAGR', 0):.1%}", "BENCHMARK": "> 20%", "VERDICT": "🏆 Elite" if metrics.get('CAGR', 0) > 0.2 else "😐 Std"},
        {"METRIC": "MAR Ratio", "YOURS": f"{metrics.get('MAR Ratio', 0):.2f}", "BENCHMARK": "> 1.0", "VERDICT": "🚀 Elite" if metrics.get('MAR Ratio', 0) > 1.0 else "😐 Std"},
        
        # --- RISK ---
        {"METRIC": "Max Drawdown", "YOURS": f"{metrics.get('Max Drawdown', 0):.1%}", "BENCHMARK": "< 15%", "VERDICT": "🛡️ Safe" if abs(metrics.get('Max Drawdown', 0)) < 0.15 else "⚠️ High Risk"},
        {"METRIC": "Recovery Time", "YOURS": f"{metrics.get('Recovery Time', 0)} Days", "BENCHMARK": "< 30 Days", "VERDICT": "⚡ Fast" if metrics.get('Recovery Time', 0) < 30 else "🐢 Slow"},
        {"METRIC": "Sharpe Ratio", "YOURS": f"{metrics.get('Sharpe Ratio', 0):.2f}", "BENCHMARK": "> 1.5", "VERDICT": "🔥 Good" if metrics.get('Sharpe Ratio', 0) > 1.5 else "😐 Std"},
        {"METRIC": "Sortino Ratio", "YOURS": f"{metrics.get('Sortino Ratio', 0):.2f}", "BENCHMARK": "> 2.0", "VERDICT": "💎 Strong" if metrics.get('Sortino Ratio', 0) > 2.0 else "😐 Std"},

        # --- CONSISTENCY ---
        {"METRIC": "Profit Factor", "YOURS": f"{metrics.get('Profit Factor', 0):.2f}", "BENCHMARK": "> 1.5", "VERDICT": "💰 Rich" if metrics.get('Profit Factor', 0) > 1.5 else "😐 Std"},
        {"METRIC": "Daily Win Rate", "YOURS": f"{metrics.get('Win Rate (Daily)', 0):.0%}", "BENCHMARK": "50-55%", "VERDICT": "✅ Stable" if metrics.get('Win Rate (Daily)', 0) > 0.5 else "🔻 Low"},
    ]
    return pd.DataFrame(data)

def calculate_institutional_score(metrics):
    """
    Calculates a weighted score (0-100) to rate the strategy's professionalism.
    Focuses on Risk-Adjusted Returns (Sharpe/MAR) over raw gains.
    """
    score = 0
    max_score = 0
    
    # 1. Sharpe Ratio (Weight: 30%) -> Insts love Sharpe > 2.0
    # Score 30 pts if Sharpe >= 2.0, scaled down if lower
    sharpe = metrics.get('Sharpe Ratio', 0)
    score += min(30, (sharpe / 2.0) * 30)
    max_score += 30
    
    # 2. MAR Ratio (Weight: 25%) -> Return / MaxDD > 1.0 is elite
    mar = metrics.get('MAR Ratio', 0)
    score += min(25, (mar / 1.0) * 25)
    max_score += 25
    
    # 3. Max Drawdown (Weight: 25%) -> Penalize heavy drawdowns
    # Full 25 pts if DD < 10%. 0 pts if DD > 30%
    dd = abs(metrics.get('Max Drawdown', 0))
    if dd < 0.10: score += 25
    elif dd < 0.20: score += 15
    elif dd < 0.30: score += 5
    max_score += 25
    
    # 4. Sortino (Weight: 20%) -> Penalize downside volatility
    sortino = metrics.get('Sortino Ratio', 0)
    score += min(20, (sortino / 3.0) * 20)
    max_score += 20
    
    return min(100, score)

def calculate_future_projections(current_equity, target_cagr):
    """
    Projects equity based on a provided CAGR rate.
    """
    # 1. Generate Target Dates
    today = pd.Timestamp.now().normalize()
    target_dates = []
    
    # A. Monthly: End of month for next 12 months
    for i in range(0, 13): 
        future_date = today + pd.tseries.offsets.MonthEnd(i)
        if future_date < today: 
            future_date = today + pd.tseries.offsets.MonthEnd(i+1)
        target_dates.append(future_date)
        
    # B. Yearly: End of [Current Month] for next 10 years
    current_month_index = today.month 
    for i in range(2, 11): 
        future_year = today.year + i
        future_dt = pd.Timestamp(year=future_year, month=current_month_index, day=1) + pd.tseries.offsets.MonthEnd(0)
        target_dates.append(future_dt)

    target_dates = sorted(list(set(target_dates)))
    
    # 2. Calculate Projections
    projections = []
    for date in target_dates:
        years_future = (date - today).days / 365.25
        # Future Value Formula: PV * (1+r)^t
        future_val = current_equity * ((1 + target_cagr) ** years_future)
        
        projections.append({
            "Date": date,
            "Timeline": "Next 12 Months" if years_future <= 1.05 else "10-Year Vision",
            "Projected Value": future_val
        })
        
    return pd.DataFrame(projections)

def calculate_3d_physics(df):
    """
    Calculates Velocity, Acceleration, and Jerk (The 3rd Derivative).
    """
    phys_df = df.copy()
    
    # 1. Velocity (Daily Return %)
    phys_df['velocity'] = phys_df['equity'].pct_change() * 100
    
    # 2. Acceleration (Change in Velocity)
    phys_df['acceleration'] = phys_df['velocity'].diff()
    
    # 3. Jerk (Change in Acceleration - The "Whiplash" factor)
    phys_df['jerk'] = phys_df['acceleration'].diff()

    # Smooth slightly to reduce noise
    phys_df['vel_smooth'] = phys_df['velocity'].rolling(3).mean()
    phys_df['acc_smooth'] = phys_df['acceleration'].rolling(3).mean()
    phys_df['jerk_smooth'] = phys_df['jerk'].rolling(3).mean()
    
    return phys_df.dropna()

def calculate_rolling_edge(df, window=30):
    r_df = df.copy()
    r_df['daily_return'] = r_df['equity'].pct_change()
    
    # --- OFFENSIVE METRICS ---
    # 30-Day Return
    r_df['rolling_return'] = r_df['equity'].pct_change(periods=window) * 100
    
    # 30-Day Sharpe
    roll_mean = r_df['daily_return'].rolling(window).mean()
    roll_std = r_df['daily_return'].rolling(window).std()
    r_df['rolling_sharpe'] = (roll_mean / roll_std) * (252 ** 0.5)
    
    # --- DEFENSIVE METRICS ---
    # 30-Day Rolling Drawdown
    rolling_peak = r_df['equity'].rolling(window=window, min_periods=1).max()
    r_df['rolling_dd'] = ((r_df['equity'] - rolling_peak) / rolling_peak) * 100

    # 30-Day Rolling Sortino
    downside_returns = r_df['daily_return'].copy()
    downside_returns[downside_returns > 0] = 0
    roll_downside_std = downside_returns.rolling(window).std()
    
    r_df['rolling_sortino'] = r_df.apply(
        lambda row: 0.0 if roll_downside_std.loc[row.name] == 0 
        else (roll_mean.loc[row.name] / roll_downside_std.loc[row.name]) * (252 ** 0.5), axis=1
    )
    
    # --- CONSISTENCY & REGIME METRICS ---
    # 30-Day Rolling Daily Win Rate (%)
    r_df['is_win'] = (r_df['daily_return'] > 0).astype(int)
    r_df['rolling_win_rate'] = r_df['is_win'].rolling(window=window).mean() * 100
    
    # 30-Day Rolling Volatility (Annualized %)
    r_df['rolling_vol'] = roll_std * (252 ** 0.5) * 100
    
    return r_df.dropna(subset=['rolling_return', 'rolling_sharpe', 'rolling_dd'])

def format_log_line(line):
    """Formats a single log line to look like VS Code syntax highlighting."""
    # 1. Safety escape for HTML
    clean_line = line.replace("<", "&lt;").replace(">", "&gt;")
    
    # 2. Colorize Timestamps (e.g., 2026-02-07 09:11:52)
    clean_line = re.sub(
        r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', 
        r'<span class="log-ts">\1</span>', 
        clean_line
    )
    
    # 3. Colorize Tags ([INFO], [ERROR], etc.)
    clean_line = clean_line.replace("[INFO]", '<span class="log-info">[INFO]</span>')
    clean_line = clean_line.replace("[WARNING]", '<span class="log-warn">[WARNING]</span>')
    clean_line = clean_line.replace("[ERROR]", '<span class="log-err">[ERROR]</span>')
    
    # 4. Colorize Tickers (e.g., [AAPL])
    clean_line = re.sub(
        r'\[([A-Z]{2,5})\]', 
        r'<span class="log-ticker">[\1]</span>', 
        clean_line
    )

    return f'<div class="log-line">{clean_line}</div>'

# === SIDEBAR CONFIG ===
with st.sidebar:
    st.header("🦅 Angel Control")
    auto_refresh = st.toggle("Enable Auto-Refresh (60s)", value=True)
    
    st.divider()
    st.subheader("🔮 Projection Tuning")
    # Allows you to override the CAGR for projections
    use_manual_cagr = st.checkbox("Manual CAGR Override")
    manual_cagr = st.slider("Target CAGR %", 0, 100, 25) / 100
    
    if st.button("Force Refresh Now", type="primary"):
        st.cache_data.clear() 
        st.rerun()

# === DASHBOARD LOGIC ===
api = init_alpaca()
if not api: st.stop()

# 1. ACCOUNT OVERVIEW
account, positions, orders = get_account_data(api)

if account:
    # Expanded to 6 columns to fit the Alpha Gauge
    col1, col2, col_alpha, col3, col_var, col4 = st.columns(6) 
    
    equity = float(account['equity'])
    last_equity = float(account['last_equity'])
    buying_power = float(account['buying_power'])
    
    daily_pl_pct = (equity - last_equity) / last_equity * 100
    daily_pl_abs = equity - last_equity
    
    # --- NEW: Alpha Calculation ---
    spy_return = get_market_benchmark()
    daily_alpha = daily_pl_pct - spy_return
    
    # Value at Risk Calculation (Your existing code)
    total_var = sum([abs(float(p['market_value'])) * 0.03 for p in positions]) if positions else 0.0
    var_pct = (total_var / equity) * 100 if equity > 0 else 0.0
    
    col1.metric("Net Liquidity", f"${equity:,.2f}", f"{daily_pl_pct:.2f}%")
    col2.metric("Day P/L", f"${daily_pl_abs:,.2f}")
    
    # --- NEW: Alpha Metric ---
    col_alpha.metric("Daily Alpha (vs SPY)", f"{daily_alpha:+.2f}%", f"SPY: {spy_return:+.2f}%", delta_color="normal")
    
    col3.metric("Buying Power", f"${buying_power:,.2f}")
    col_var.metric("Open Risk (VaR)", f"${total_var:,.2f}", f"-{var_pct:.2f}% Eq", delta_color="inverse")
    
    # Process Logs
    logs = read_bot_logs()
    last_run_str, last_run_dt, parsed_signals, watchlist_data, conviction_data = parse_latest_run_logic(logs)

    # Calculate "Time Since Last Run"
    status_label = "Bot Status"
    status_val = "Unknown"

    if last_run_dt:
        # Streamlit server time vs Log time safety alignment
        diff = datetime.now() - last_run_dt 
        seconds_ago = int(diff.total_seconds())
        minutes_ago = int(seconds_ago / 60)
        
        if minutes_ago < 10:
            status_val = "🟢 Active"
        elif minutes_ago < 60:
            status_val = f"🟡 Idle ({minutes_ago}m)"
        else:
            status_val = f"🔴 Stale ({int(minutes_ago/60)}h)"
    
    col4.metric(status_label, status_val, delta=f"Last Log: {last_run_str}", delta_color="off")
    
    # --- ADDED: BOT HEARTBEAT COUNTDOWN ---
    if status_val == "🟢 Active" and seconds_ago < 300:
        safe_seconds_ago = max(0, seconds_ago) 
        seconds_left = max(0, 300 - safe_seconds_ago)
        
        # FIX: Convert to strict integer 0-100 to prevent Streamlit float exceptions
        progress_val = int(max(0, min(100, (safe_seconds_ago / 300.0) * 100)))
        
        st.progress(progress_val, text=f"⏳ Next Market Scan in ~{seconds_left}s")

st.divider()

# 2. MAIN CONTENT TABS
# --- ADDED: PENDING / STUCK ORDER ALERTS ---
if isinstance(orders, list):
    pending_orders = [o for o in orders if isinstance(o, dict) and o.get('status') in ['new', 'accepted', 'partially_filled', 'pending_new']]
    for po in pending_orders:
        created_at = po.get('created_at')
        if created_at:
            try:
                created_dt = pd.to_datetime(created_at).tz_convert('UTC')
                now_dt = pd.Timestamp.now(tz='UTC')
                seconds_open = max(0, (now_dt - created_dt).total_seconds())
                
                side_str = po.get('side', 'UNKNOWN').upper()
                qty_str = po.get('qty', '?')
                sym_str = po.get('symbol', '?')
                
                if seconds_open > 60:
                    st.error(f"⚠️ **Execution Alert:** {side_str} order for {qty_str} {sym_str} has been pending for {int(seconds_open)}s! High slippage risk.")
                else:
                    st.info(f"🔄 **Transmitting:** {side_str} {qty_str} {sym_str} (Routing to market: {int(seconds_open)}s ago)")
            except Exception:
                pass

# 2. MAIN CONTENT TABS
tab1, tab2, tab3 = st.tabs(["🧠 Bot Logic & Positions", "📜 Raw Logs", "📈 Performance"])

with tab1:
    # --- 1. MARKET PULSE ---
    avg_market_move = 0.0
    if positions:
        avg_market_move = sum([float(p['unrealized_plpc']) for p in positions]) * 100
        sentiment_score = max(0.0, min(1.0, 0.5 + (avg_market_move / 5)))
    else:
        sentiment_score = 0.5

    st.markdown("### 🌡️ Market Pulse")
    s_col1, s_col2, s_col3 = st.columns([3, 1, 6]) # The empty 3rd column acts as a spacer
    with s_col1:
        st.progress(int(max(0, min(100, sentiment_score * 100))))
    with s_col2:
        if avg_market_move > 0.5: st.success("BULLISH")
        elif avg_market_move < -0.5: st.error("BEARISH")
        else: st.warning("NEUTRAL")

    # --- UPGRADED: CAPITAL DEPLOYMENT STATES ---
    st.markdown("#### 🔋 Capital Deployment Status")
    
    # Calculate exactly how much cash is locked in positions
    active_capital = sum([abs(float(p['market_value'])) for p in positions]) if positions else 0.0
    cash_capital = equity - active_capital  # <-- Uses the corrected cash calculation
    total_capital = active_capital + cash_capital
    
    # Calculate percentages
    active_pct = (active_capital / total_capital * 100) if total_capital > 0 else 0
    cash_pct = (cash_capital / total_capital * 100) if total_capital > 0 else 100
    
    bot_states = extract_bot_states(logs)
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("💼 Active Capital", f"${active_capital:,.2f}", f"{active_pct:.1f}% Deployed", delta_color="off")
    sc2.metric("💵 Dry Powder", f"${cash_capital:,.2f}", f"{cash_pct:.1f}% Cash", delta_color="off")
    
    # FIX: Hardcoded the ticker list so it doesn't look for the missing 'config' variable
    monitored_tickers = ['IONQ', 'KO', 'OXY', 'BAC', 'GM', 'PFE', 'PYPL','FCX']
    sc3.metric("🤖 Active Agents", f"{len(positions)} / {len(monitored_tickers)}")

    # --- ADDED: NEURAL SKEW / MACRO BIAS ---
    if parsed_signals:
        long_count = sum(1 for s in parsed_signals.values() if "Long" in s)
        short_count = sum(1 for s in parsed_signals.values() if "Short" in s)
        hold_count = len(parsed_signals) - long_count - short_count
        
        st.markdown("#### ⚖️ Bot Macro Bias (Neural Skew)")
        # Normalize for progress bar (0.0 to 1.0)
        total_signals = len(parsed_signals)
        skew_val = (long_count + (hold_count * 0.5)) / total_signals if total_signals > 0 else 0.5
        
        st.progress(int(max(0, min(100, skew_val * 100))))
        b1, b2, b3 = st.columns(3)
        b1.caption(f"🟢 Long Bias: {long_count}")
        b2.caption(f"⚪ Neutral/Hold: {hold_count}")
        b3.caption(f"🔴 Short Bias: {short_count}")

    st.divider()

    # --- 2. NEURAL CONVICTION RADAR ---
    st.subheader("🧠 Neural Conviction Levels")
    if conviction_data:
        # Convert nested dictionary to flat DataFrame
        flat_data = [
            {"Ticker": t, "Confidence": d["Confidence"], "Action": d["Action"]} 
            for t, d in conviction_data.items()
        ]
        df_conv = pd.DataFrame(flat_data)
        df_conv = df_conv.sort_values(by='Confidence', ascending=False)
        
        # Create Chart text combining Action and Confidence
        df_conv['Chart_Text'] = df_conv.apply(lambda row: f"{row['Action']}<br>{row['Confidence']:.1f}%" if row['Action'] else f"{row['Confidence']:.1f}%", axis=1)

        fig_conf = px.bar(
            df_conv, 
            x='Ticker', 
            y='Confidence', 
            color='Confidence',
            color_continuous_scale=['#4a1c1c', '#ffb000', '#00ff41'], 
            range_y=[0, 100],
            text='Chart_Text' # <--- This puts the Action State on the bar
        )
        
        fig_conf.update_traces(textposition='inside', textfont_size=14, textfont_color='white')
        
        fig_conf.update_layout(
            height=150, 
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title=None, 
            yaxis_title="Confidence %",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#cccccc'},
            yaxis=dict(showgrid=True, gridcolor='#333'),
            xaxis=dict(showgrid=False, categoryorder='total descending') 
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Waiting for first model run to populate conviction data...")

    st.divider()

# --- 3. MAIN COLUMNS ---
    c1, c2 = st.columns([3, 4])
    
    with c1:
        # --- UPGRADED: 4 Tabs ---
        tab_wl, tab_log, tab_health, tab_edge = st.tabs(["🔭 Watchlist", "📝 Decisions", "🖥️ Risk & Telemetry", "🔪 Execution & Edge"])
        
        with tab_wl:
            if watchlist_data:
                wl_df = pd.DataFrame(watchlist_data)
                st.dataframe(wl_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No high-confidence setups detected yet.")

        with tab_log:
            if parsed_signals:
                sig_df = pd.DataFrame(list(parsed_signals.items()), columns=["Ticker", "Decision"])
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            else:
                st.info("No signals parsed from recent logs.")
                
        with tab_health:
            st.markdown("#### Server & API Telemetry")
            cpu, ram, ping = get_system_telemetry()
            
            t1, t2, t3 = st.columns(3)
            t1.metric("CPU Load", f"{cpu}%", delta="High" if cpu > 80 else "Normal", delta_color="inverse")
            t2.metric("RAM Util", f"{ram}%", delta="High" if ram > 85 else "Normal", delta_color="inverse")
            t3.metric("API Latency", f"{ping}ms", delta="Lag" if ping > 300 else "Fast", delta_color="inverse")

            st.divider()
            
            st.markdown("#### Margin Distance")
            maint_margin = float(account.get('maintenance_margin', 0)) if account else 0.0
            margin_util = (maint_margin / equity * 100) if equity > 0 else 0.0
            st.progress(int(max(0, min(100, margin_util))), text=f"Margin Capacity Used: {margin_util:.1f}%")
            if margin_util > 80:
                st.error("⚠️ CRITICAL: Approaching Maintenance Margin Call!")

            st.divider()

            st.markdown("#### Active Position Correlation")
            if positions and len(positions) > 1:
                active_tickers = [p['symbol'] for p in positions]
                corr_matrix = get_correlation_matrix(active_tickers)
                
                if corr_matrix is not None:
                    fig_corr = px.imshow(
                        corr_matrix, 
                        text_auto=".2f", 
                        color_continuous_scale="RdBu_r", 
                        zmin=-1, zmax=1
                    )
                    fig_corr.update_layout(
                        height=280, 
                        margin=dict(l=0, r=0, t=10, b=0), 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        font={'color': '#cccccc'}
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.caption("Need at least 2 active positions to plot correlation.")

        # --- NEW TAB: EXECUTION & EDGE ---
        with tab_edge:
            st.markdown("#### ⚖️ Edge Quality")
            
            # Fetch from previously calculated metrics
            sqn_val = metrics.get('SQN', 0)
            ulcer_val = metrics.get('Ulcer Index', 0)
            
            e1, e2 = st.columns(2)
            e1.metric("System Quality No. (SQN)", f"{sqn_val:.2f}", delta="Robust" if sqn_val > 1.5 else "Weak", delta_color="normal")
            e2.metric("Ulcer Index (Pain)", f"{ulcer_val:.2f}", delta="Safe" if ulcer_val < 5.0 else "Stressful", delta_color="inverse")
            
            st.divider()
            st.markdown("#### 🎯 Excursion Analysis (MAE vs MFE)")
            st.caption("Scatter plot of recent closed trades. Identifies if stops are too tight or winners are choked.")
            
            df_ex = get_trade_excursions(api, orders)
            
            if not df_ex.empty:
                fig_ex = px.scatter(
                    df_ex, x="MAE (%)", y="MFE (%)", color="Result",
                    hover_data=["Ticker", "PnL (%)", "Type"],
                    color_discrete_map={"Win": "#00ff41", "Loss": "#ff4b4b"}
                )
                
                # Add crosshairs for standard Stop Loss / Take Profit boundaries
                fig_ex.add_vline(x=-3.0, line_dash="dash", line_color="red", annotation_text="Hard Stop (-3%)", annotation_position="top right")
                fig_ex.add_hline(y=6.0, line_dash="dash", line_color="green", annotation_text="Standard TP (+6%)", annotation_position="bottom right")
                
                fig_ex.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                fig_ex.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#cccccc'},
                    xaxis=dict(title="Max Adverse Excursion (Pain %)", showgrid=True, gridcolor='#333', zerolinecolor='white'),
                    yaxis=dict(title="Max Favorable (Gain %)", showgrid=True, gridcolor='#333', zerolinecolor='white')
                )
                st.plotly_chart(fig_ex, use_container_width=True)
            else:
                st.info("Gathering excursion data. Close more trades to populate scatter plot.")

            st.divider()
            
            # --- MOVED RECENT ORDERS & SLIPPAGE HERE ---
            st.markdown("#### 📜 Recent Slippage")
            today_utc = pd.Timestamp.now(tz='UTC').date()
            if isinstance(orders, list):
                order_data = []
                for o in orders[:8]: # Show last 8
                    if isinstance(o, dict) and o.get('status') == 'filled':
                        t = o.get('filled_at', '')
                        t_fmt = t[5:16].replace('T', ' ') if len(t) >= 16 else t
                        
                        limit_price = float(o.get('limit_price', 0)) if o.get('limit_price') else 0.0
                        fill_price = float(o.get('filled_avg_price', 0)) if o.get('filled_avg_price') else 0.0
                        
                        slippage = 0.0
                        if limit_price > 0 and fill_price > 0:
                            if o.get('side') == 'buy':
                                slippage = ((fill_price - limit_price) / limit_price) * 100
                            else:
                                slippage = ((limit_price - fill_price) / limit_price) * 100

                        order_data.append({
                            "Ticker": o.get('symbol', 'N/A'),
                            "Side": o.get('side', 'N/A').upper(),
                            "Fill Price": f"${fill_price:.2f}",
                            "Slippage": f"{slippage:+.2f}%" if limit_price > 0 else "N/A (MKT)"
                        })

                if order_data:
                    df_orders = pd.DataFrame(order_data)
                    def highlight_slippage(val):
                        if isinstance(val, str) and "%" in val:
                            num = float(val.replace("%", "").replace("+", ""))
                            if num > 0: return 'color: #ff4b4b' 
                            if num < 0: return 'color: #00ff41' 
                        return ''
                    
                    def highlight_side(val):
                        if val == 'BUY': return 'color: #00ff41; font-weight: bold;'
                        if val == 'SELL': return 'color: #ff4b4b; font-weight: bold;'
                        return ''

                    styled_df = (df_orders.style
                                 .map(highlight_slippage, subset=['Slippage'])
                                 .map(highlight_side, subset=['Side']))

                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No recent filled orders found.")

    with c2:
        st.subheader("💼 Capital & Active Portfolio")
        
        # --- UPGRADED: CAPITAL ALLOCATION DONUT CHART ---
        # Uses the 'cash_capital' calculated at the top of Tab 1 (fixes the 78% bug)
        allocation_data = [{"Asset": "CASH", "Value": cash_capital}]
        for p in positions:
            allocation_data.append({"Asset": p['symbol'], "Value": abs(float(p['market_value']))})
        
        if allocation_data:
            fig_alloc = px.pie(
                pd.DataFrame(allocation_data), values='Value', names='Asset', hole=0.65,
                color_discrete_sequence=['#2d2d2d'] + px.colors.qualitative.Pastel
            )
            fig_alloc.update_layout(
                margin=dict(l=0, r=0, t=10, b=10), height=220,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#cccccc'}, showlegend=True,
                legend=dict(orientation="v", yanchor="auto", y=0.5, xanchor="left", x=1.0)
            )
            fig_alloc.add_annotation(text=f"Total Eq<br>${equity:,.0f}", x=0.5, y=0.5, font_size=14, showarrow=False)
            st.plotly_chart(fig_alloc, use_container_width=True)
            
            # --- ADDED: NEXT SLOT DEPLOYMENT ESTIMATE ---
            monitored_tickers = ['IONQ', 'KO', 'OXY', 'BAC', 'GM', 'PFE', 'PYPL', 'FCX']
            est_slot_size = equity / len(monitored_tickers)
            st.caption(f"🤖 **Bot Pre-Auth:** Estimated next trade size is **~${est_slot_size:,.2f}** per signal.")
            
            # --- ADDED: SECTOR / INDEX EXPOSURE ---
            ASSET_INDEX_MAP = {
                'MARA': 'BLOK', 'PLTR': 'IGV', 'SOFI': 'XLF', 'HOOD': 'XLF',
                'INTC': 'SOXX', 'IONQ': 'QTUM', 'OXY': 'XLE', 'PYPL': 'XLK',
                'CSCO': 'XLK', 'HPE': 'XLK', 'F': 'XLY', 'BAC': 'XLF',
                'KO': 'XLP', 'KR': 'XLP', 'PFE': 'XLV', 'GM': 'XLY', 'FCX': 'XLB', 
            }
            sector_data = {}
            for p in positions:
                sec = ASSET_INDEX_MAP.get(p['symbol'], 'Other')
                sector_data[sec] = sector_data.get(sec, 0) + abs(float(p['market_value']))
            
            if sector_data:
                df_sec = pd.DataFrame(list(sector_data.items()), columns=['Index', 'Exposure']).sort_values('Exposure', ascending=True)
                fig_sec = px.bar(df_sec, x='Exposure', y='Index', orientation='h', text_auto='$.0f')
                fig_sec.update_traces(marker_color='#569cd6', textposition='inside')
                fig_sec.update_layout(
                    height=120 + (len(df_sec) * 20), margin=dict(l=0, r=0, t=25, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#cccccc'}, xaxis_visible=False,
                    title=dict(text="Risk by Mapped Index", font=dict(size=14))
                )
                st.plotly_chart(fig_sec, use_container_width=True)

        # --- UPGRADED PORTFOLIO TABLE ---
        if positions:
            pos_data = []
            for p in positions:
                sym = p['symbol']
                side = p['side'].lower()
                entry = float(p['avg_entry_price'])
                current = float(p['current_price'])
                qty = abs(float(p['qty']))
                
                # Calculate Invested Amount
                invested_amt = entry * qty
                
                # Calculate Journey to TP (0.0 to 1.0)
                if side == 'long':
                    sl, tp = entry * 0.97, entry * 1.05
                    progress = max(0.0, min(1.0, (current - sl) / (tp - sl)))
                else:
                    sl, tp = entry * 1.03, entry * 0.95
                    progress = max(0.0, min(1.0, (sl - current) / (sl - tp)))

                # Calculate Days Held (Max 5)
                days_held = 0
                if isinstance(orders, list):
                    for o in orders:
                        if isinstance(o, dict) and o.get('symbol') == sym and o.get('status') == 'filled':
                            filled_at = o.get('filled_at')
                            if filled_at:
                                try:
                                    filled_dt = pd.to_datetime(filled_at).tz_convert('UTC')
                                    now_dt = pd.Timestamp.now(tz='UTC')
                                    days_held = max(0, (now_dt - filled_dt).days)
                                except Exception:
                                    pass
                            break

                pos_data.append({
                    "Ticker": sym, 
                    "Side": side.upper(),
                    "Invested": invested_amt, # <--- NEW
                    "Qty": qty,
                    "P/L (%)": float(p['unrealized_plpc']) * 100,
                    "Journey": progress,
                    "Days Held": f"{days_held}/5"
                })
            
            st.dataframe(
                pd.DataFrame(pos_data),
                use_container_width=True,
                column_config={
                    "Invested": st.column_config.NumberColumn("Invested", format="$%.2f"), # <--- FORMATTED AS CURRENCY
                    "P/L (%)": st.column_config.NumberColumn("P/L (%)", format="%.2f%%"),
                    "Journey": st.column_config.ProgressColumn(
                        "Journey to TP", help="Green bar moving right towards Take Profit.",
                        min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
                hide_index=True
            )
            # --- UPGRADED: THE FLASHPOINT ALERT (TRUE R-MULTIPLES) ---
            st.markdown("##### 🎯 Immediate Flashpoints (True R-Multiple)")
            closest_tp, closest_sl = None, None
            max_r, min_r = -999.0, 999.0

            for p_data in pos_data:
                # Calculate True R: (Current PnL %) / (Stop Loss %)
                # Assumes standard 3% SL from bot config
                sl_pct = 3.0 
                true_r = p_data["P/L (%)"] / sl_pct
                
                # Find the trade closest to Take Profit (Highest +R)
                if true_r > max_r:
                    max_r = true_r
                    closest_tp = p_data["Ticker"]
                    
                # Find the trade closest to Stop Loss (Lowest -R)
                if true_r < min_r:
                    min_r = true_r
                    closest_sl = p_data["Ticker"]

            f1, f2 = st.columns(2)
            if closest_tp and max_r > 0: 
                f1.success(f"🟢 **Highest R:** {closest_tp} (Floating: +{max_r:.2f}R)")
            if closest_sl and min_r < 0: 
                f2.error(f"🔴 **Lowest R:** {closest_sl} (Floating: {min_r:.2f}R)")
                
        else:
            st.caption("No active positions currently held.")

        # --- UPGRADED: RECENT ORDERS & SLIPPAGE ---
        st.divider()

        today_utc = pd.Timestamp.now(tz='UTC').date()
        if isinstance(orders, list):
            trades_today = sum(1 for o in orders if isinstance(o, dict) and o.get('status') == 'filled' and pd.to_datetime(o.get('filled_at')).tz_convert('UTC').date() == today_utc)
        else:
            trades_today = 0

        c_ord1, c_ord2 = st.columns([3, 1])
        c_ord1.subheader("📜 Recent Fills & Execution Quality")
        if trades_today > 4:
            c_ord2.error(f"⚠️ Trades Today: {trades_today}")
        else:
            c_ord2.info(f"⚡ Trades Today: {trades_today}")

        if orders and isinstance(orders, list):
            order_data = []
            for o in orders[:5]: 
                if isinstance(o, dict) and o.get('status') == 'filled':
                    t = o.get('filled_at', '')
                    t_fmt = t[5:16].replace('T', ' ') if len(t) >= 16 else t
                    
                    # Calculate Slippage if it was a Limit order that filled
                    limit_price = float(o.get('limit_price', 0)) if o.get('limit_price') else 0.0
                    fill_price = float(o.get('filled_avg_price', 0)) if o.get('filled_avg_price') else 0.0
                    
                    slippage = 0.0
                    if limit_price > 0 and fill_price > 0:
                        if o.get('side') == 'buy':
                            slippage = ((fill_price - limit_price) / limit_price) * 100
                        else:
                            slippage = ((limit_price - fill_price) / limit_price) * 100

                    order_data.append({
                        "Time": t_fmt,
                        "Ticker": o.get('symbol', 'N/A'),
                        "Side": o.get('side', 'N/A').upper(),
                        "Qty": o.get('filled_qty', '0'),
                        "Fill Price": f"${fill_price:.2f}",
                        "Slippage": f"{slippage:+.2f}%" if limit_price > 0 else "N/A (MKT)"
                    })

            if order_data:
                df_orders = pd.DataFrame(order_data)
                
                # Apply conditional formatting to the slippage column
                def highlight_slippage(val):
                    if isinstance(val, str) and "%" in val:
                        num = float(val.replace("%", "").replace("+", ""))
                        if num > 0: return 'color: #ff4b4b' # Red for bad slippage
                        if num < 0: return 'color: #00ff41' # Green for price improvement
                    return ''
                
                # NEW: Color code the BUY/SELL side
                def highlight_side(val):
                    if val == 'BUY': return 'color: #00ff41; font-weight: bold;'
                    if val == 'SELL': return 'color: #ff4b4b; font-weight: bold;'
                    return ''

                # Chain the mappings together
                styled_df = (df_orders.style
                             .map(highlight_slippage, subset=['Slippage'])
                             .map(highlight_side, subset=['Side']))

                st.dataframe(styled_df, width="stretch", hide_index=True)
            else:
                st.caption("No recent filled orders found.")
        else:
            st.caption("No recent filled orders found.")

with tab2:
    st.markdown("### Terminal Output (Last 50 Lines)")
    
    if logs:
        recent_logs = logs[-50:] 
        formatted_logs = [format_log_line(line) for line in recent_logs]
        log_html = "".join(formatted_logs)
        st.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.write("No logs found.")

with tab3:
    # 1. Get History
    hist_df_raw = get_portfolio_history(api)
    
    if not hist_df_raw.empty and account:
        current_equity_raw = float(account['equity'])

        # === FIX: Ensure everything is UTC for comparison ===
        if hist_df_raw['timestamp'].dt.tz is None:
            hist_df_raw['timestamp'] = hist_df_raw['timestamp'].dt.tz_localize('UTC')
        
        now_ts = pd.Timestamp.now(tz='UTC') 

        # Append LIVE Raw Data
        live_row = pd.DataFrame([{
            'timestamp': now_ts, 
            'equity': current_equity_raw
        }])
        hist_df_raw = pd.concat([hist_df_raw, live_row], ignore_index=True)

        # === DATA FORK: Create "Adjusted" Copy for Metrics Only ===
        hist_df_adj = hist_df_raw.copy()
        
        # Helper to ensure mask comparison is apples-to-apples
        def apply_deposit(df, date_str, amount):
            ts = pd.Timestamp(date_str, tz='UTC')
            mask = df['timestamp'] >= ts
            df.loc[mask, 'equity'] -= amount
            return df

        # Apply all deposits strictly
        hist_df_adj = apply_deposit(hist_df_adj, "2026-01-24", 68.10)
        hist_df_adj = apply_deposit(hist_df_adj, "2026-02-12", 69.81)
        hist_df_adj = apply_deposit(hist_df_adj, "2026-02-16", 139.75)
        hist_df_adj = apply_deposit(hist_df_adj, "2026-02-26", 69.71)
        hist_df_adj = apply_deposit(hist_df_adj, "2026-03-04", 68.84)
        hist_df_adj = apply_deposit(hist_df_adj, "2026-03-13", 69.61)
        hist_df_adj = apply_deposit(hist_df_adj, "2026-03-21", 69.01)        

        # ==========================================================
        
        # --- CALCULATIONS ---
        
        # A. METRICS: Use ADJUSTED Data (Honest Strategy Score)
        metrics = calculate_advanced_metrics(hist_df_adj)
        scorecard_df = create_scorecard_df(metrics)
        inst_score = calculate_institutional_score(metrics)
        
        # Capture the honest CAGR
        valid_cagr = metrics.get("CAGR", 0.0)
        
        # B. VISUALS: Use RAW Data
        dd_df = calculate_drawdown(hist_df_raw) 
        phys_df = calculate_3d_physics(hist_df_raw) 
        day_stats, monthly_stats = calculate_seasonality(hist_df_raw)
        
        # C. PROJECTIONS: Use valid_cagr (or manual) applied to Real Money
        projection_rate = manual_cagr if use_manual_cagr else valid_cagr
        
        # Calculate projection
        proj_df = calculate_future_projections(current_equity_raw, projection_rate)

        # --- SECTION 1: THE INSTITUTIONAL GAUGE ---
        col_gauge, col_scorecard = st.columns([1, 2.5])
        
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = inst_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Strategy Grade", 'font': {'size': 20, 'color': '#e0e0e0'}},
                number = {'suffix': "/100", 'font': {'color': '#e0e0e0'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                    'bar': {'color': "#00ff41" if inst_score > 80 else "#ffb000"},
                    'bgcolor': "#1e1e1e",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.3)'},
                        {'range': [50, 80], 'color': 'rgba(255, 176, 0, 0.3)'},
                        {'range': [80, 100], 'color': 'rgba(0, 255, 65, 0.3)'}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': inst_score}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if inst_score > 80:
                st.markdown("<div style='text-align: center; color: #00ff41; font-weight: bold;'>🚀 INSTITUTIONAL GRADE</div>", unsafe_allow_html=True)
            elif inst_score > 50:
                st.markdown("<div style='text-align: center; color: #ffb000; font-weight: bold;'>⚡ PROFESSIONAL RETAIL</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center; color: #ff4b4b; font-weight: bold;'>🎲 DEGEN / RETAIL</div>", unsafe_allow_html=True)

        with col_scorecard:
            st.markdown("### 📊 Metrics Breakdown (Adj. for Deposits)")
            st.dataframe(
                scorecard_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "METRIC": st.column_config.TextColumn("Metric", width="medium"),
                    "YOURS": st.column_config.TextColumn("Your Bot", width="small"),
                    "BENCHMARK": st.column_config.TextColumn("Target", width="small"),
                    "VERDICT": st.column_config.TextColumn("Verdict", width="small"),
                },
                height=280
            )

        st.divider()

        # --- SECTION 2: CHARTS (USING RAW DATA) ---
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.markdown(f"### 📈 Real Equity Curve (${current_equity_raw:,.2f})")
            
            # Using RAW DF for the chart
            max_equity = hist_df_raw['equity'].max()
            fig_eq = px.area(hist_df_raw, x='timestamp', y='equity')
            fig_eq.update_traces(line_color='#00ff41', fillcolor='rgba(0, 255, 65, 0.1)')
            fig_eq.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None,
                yaxis_title=None,
                showlegend=False,
                height=300,
                yaxis=dict(range=[hist_df_raw['equity'].min() * 0.95, max_equity * 1.02], rangemode="normal")
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_perf2:
            st.markdown("### 📉 Real Risk (Drawdown)")
            # Using RAW DF (Drawdowns will look smaller relative to new higher peaks)
            fig_dd = px.area(dd_df, x='timestamp', y='drawdown')
            fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
            fig_dd.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None, showlegend=False, height=300, yaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_dd, use_container_width=True)

        # --- NEW SECTION: LONG VS SHORT ATTRIBUTION ---
        st.divider()
        st.subheader("⚔️ Long vs. Short Attribution")
        
        if isinstance(orders, list) and len(orders) > 0:
            long_wins, long_losses = 0, 0
            short_wins, short_losses = 0, 0
            
            # Simple heuristic: Look at realized PnL of closed legs
            for pos in positions:
                # Currently active positions (unrealized)
                if pos['side'] == 'long':
                    if float(pos['unrealized_pl']) > 0: long_wins += 1
                    else: long_losses += 1
                elif pos['side'] == 'short':
                    if float(pos['unrealized_pl']) > 0: short_wins += 1
                    else: short_losses += 1

            total_longs = long_wins + long_losses
            total_shorts = short_wins + short_losses
            
            long_wr = (long_wins / total_longs * 100) if total_longs > 0 else 0
            short_wr = (short_wins / total_shorts * 100) if total_shorts > 0 else 0
            
            c_ls1, c_ls2, _spacer = st.columns([1, 1, 4])
            c_ls1.metric("🟢 Long Win Rate (Active)", f"{long_wr:.1f}%", f"{total_longs} positions", delta_color="off")
            c_ls2.metric("🔴 Short Win Rate (Active)", f"{short_wr:.1f}%", f"{total_shorts} positions", delta_color="off")
            st.caption("*Note: Displays active state. Full historical attribution requires database integration.*")

        # --- NEW SECTION: ROLLING EDGE ---
        st.divider()
        st.markdown("### 🔄 30-Day Rolling Edge (Momentum, Defense & Regime)")
        
        roll_df = calculate_rolling_edge(hist_df_adj, window=30)
        
        if not roll_df.empty:
            # Create a 3x2 grid
            c_roll1, c_roll2 = st.columns(2)
            c_roll3, c_roll4 = st.columns(2)
            c_roll5, c_roll6 = st.columns(2)
            
            with c_roll1:
                st.caption("30-Day Rolling Return (%)")
                fig_roll_ret = px.area(roll_df, x='timestamp', y='rolling_return')
                fig_roll_ret.update_traces(line_color='#569cd6', fillcolor='rgba(86, 156, 214, 0.2)')
                fig_roll_ret.add_hline(y=0, line_dash="dash", line_color="white")
                fig_roll_ret.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_ret, use_container_width=True)

            with c_roll2:
                st.caption("30-Day Rolling Sharpe Ratio")
                fig_roll_shp = px.line(roll_df, x='timestamp', y='rolling_sharpe')
                fig_roll_shp.update_traces(line_color='#c586c0')
                fig_roll_shp.add_hline(y=1.5, line_dash="dot", line_color="#00ff41", annotation_text="Elite Target")
                fig_roll_shp.add_hline(y=0, line_dash="dash", line_color="#ff4b4b")
                fig_roll_shp.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_shp, use_container_width=True)

            with c_roll3:
                st.caption("30-Day Rolling Max Drawdown (%)")
                fig_roll_dd = px.area(roll_df, x='timestamp', y='rolling_dd')
                fig_roll_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
                fig_roll_dd.add_hline(y=-5.0, line_dash="dot", line_color="#ffb000", annotation_text="Pain Threshold")
                fig_roll_dd.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_dd, use_container_width=True)

            with c_roll4:
                st.caption("30-Day Rolling Sortino Ratio")
                fig_roll_srt = px.line(roll_df, x='timestamp', y='rolling_sortino')
                fig_roll_srt.update_traces(line_color='#cca700') 
                fig_roll_srt.add_hline(y=2.0, line_dash="dot", line_color="#00ff41", annotation_text="Elite Target")
                fig_roll_srt.add_hline(y=0, line_dash="dash", line_color="#ff4b4b")
                fig_roll_srt.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_srt, use_container_width=True)
                
            with c_roll5:
                st.caption("30-Day Rolling Daily Win Rate (%)")
                # Using a bar chart to represent daily consistency
                fig_roll_win = px.bar(roll_df, x='timestamp', y='rolling_win_rate')
                fig_roll_win.update_traces(marker_color='#4CAF50', opacity=0.7)
                fig_roll_win.add_hline(y=50.0, line_dash="dash", line_color="#ffb000", annotation_text="Breakeven 50%")
                fig_roll_win.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_roll_win, use_container_width=True)

            with c_roll6:
                st.caption("30-Day Rolling Volatility (Annualized %)")
                fig_roll_vol = px.line(roll_df, x='timestamp', y='rolling_vol')
                fig_roll_vol.update_traces(line_color='#ff9800')
                fig_roll_vol.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=220, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_roll_vol, use_container_width=True)
        else:
            st.caption("Not enough data yet for 30-Day Rolling metrics.")

        # --- SECTION 3: TIME INTELLIGENCE ---
        st.divider()
        st.subheader("⏳ Time Intelligence (Seasonality)")
        st.caption("Bars = Average Return (Left Axis). Lines = Win Rate % (Right Axis).")
        
        c_time1, c_time2 = st.columns(2)
        
        with c_time1:
            st.markdown("**📅 Day of Week**")
            fig_dow = go.Figure()
            fig_dow.add_trace(go.Bar(
                x=day_stats.index,
                y=day_stats['Avg_Return'],
                name='Avg Return',
                marker_color=day_stats['Avg_Return'].apply(lambda x: '#00ff41' if x >= 0 else '#ff4b4b'),
                yaxis='y1'
            ))
            fig_dow.add_trace(go.Scatter(
                x=day_stats.index,
                y=day_stats['Win_Rate'],
                name='Win Rate %',
                mode='lines+markers+text',
                text=day_stats['Win_Rate'].apply(lambda x: f"{x:.0f}%"),
                textposition="top center",
                line=dict(color='#ffb000', width=3),
                yaxis='y2'
            ))
            # For BOTH fig_dow and fig_moy, update the height and colors
            fig_dow.update_layout(
                yaxis=dict(title="Avg Return (%)", showgrid=True, gridcolor='#333'),
                yaxis2=dict(title="Win Rate (%)", overlaying='y', side='right', range=[0, 110], showgrid=False),
                showlegend=False,
                height=220, # <--- REDUCE FROM 350 to 220
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cccccc")
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        with c_time2:
            st.markdown("**🗓️ Month of Year**")
            fig_moy = go.Figure()
            fig_moy.add_trace(go.Bar(
                x=monthly_stats.index,
                y=monthly_stats['Avg_Return'],
                name='Avg Return',
                marker_color=monthly_stats['Avg_Return'].apply(lambda x: '#00ff41' if x >= 0 else '#ff4b4b'),
                yaxis='y1'
            ))
            fig_moy.add_trace(go.Scatter(
                x=monthly_stats.index,
                y=monthly_stats['Win_Rate'],
                name='Win Rate %',
                mode='lines+markers+text',
                text=monthly_stats['Win_Rate'].apply(lambda x: f"{x:.0f}%"),
                textposition="top center",
                line=dict(color='#ffb000', width=3),
                yaxis='y2'
            ))
            fig_moy.update_layout(
                yaxis=dict(title="Avg Return (%)", showgrid=True, gridcolor='#333'),
                yaxis2=dict(title="Win Rate (%)", overlaying='y', side='right', range=[0, 110], showgrid=False),
                showlegend=False,
                height=220,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cccccc")
            )
            st.plotly_chart(fig_moy, use_container_width=True)

        # --- SECTION 4: 3D PHYSICS LAB ---
        st.divider()
        st.subheader("🧊 Angel 3D Trajectory (Phase Space)")
        st.info("Dimensions: **X**=Time, **Y**=Velocity (Return), **Z**=Acceleration. **Color/Size** = JERK (Market Shock).")

        if not phys_df.empty:
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=phys_df['timestamp'],
                y=phys_df['velocity'],      
                z=phys_df['acceleration'],  
                mode='lines+markers',
                marker=dict(
                    size=abs(phys_df['jerk']) * 5 + 2, 
                    color=phys_df['jerk'],             
                    colorscale='Turbo',
                    opacity=0.8,
                    colorbar=dict(title="Jerk")
                ),
                line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
                hovertemplate = '<b>Date</b>: %{x|%Y-%m-%d}<br><b>Vel</b>: %{y:.2f}%<br><b>Acc</b>: %{z:.2f}%<br><b>Jerk</b>: %{marker.color:.2f}<extra></extra>'
            )])

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Time',
                    yaxis_title='Velocity',
                    zaxis_title='Accel',
                    xaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True),
                    yaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True),
                    zaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#333", showbackground=True),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#cccccc"),
                margin=dict(l=0, r=0, t=0, b=0),
                height=400 
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Not enough data points for Physics analysis.")

        # --- SECTION 5: FUTURE PROJECTIONS ---
        st.divider()
        
        # Determine which CAGR to use for the projection
        projection_rate = manual_cagr if use_manual_cagr else valid_cagr
        proj_label = "Manual" if use_manual_cagr else "Adj."
        
        # Calculate projection
        proj_df = calculate_future_projections(current_equity_raw, projection_rate)
        
        st.markdown(f"### 🔮 Future Projections (Based on {proj_label} CAGR: {projection_rate:.1%})")
        
        if not proj_df.empty:
            c_p1, c_p2 = st.columns([2, 1])
            with c_p1:
                fig_proj = px.line(proj_df, x='Date', y='Projected Value', markers=True, color='Timeline',
                                   color_discrete_map={"Next 12 Months": "#569cd6", "10-Year Vision": "#c586c0"})
                fig_proj.update_traces(line_width=3)
                fig_proj.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0), 
                    xaxis_title=None, 
                    yaxis_title=None, 
                    height=400, 
                    template="plotly_dark",
                    legend=dict(orientation="h", y=1.1, x=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_proj, use_container_width=True)
            with c_p2:
                # Highlight the end goal
                final_val = proj_df['Projected Value'].iloc[-1]
                st.metric("10-Year Target", f"${final_val:,.2f}", f"{projection_rate:.1%} Rate")
                
                st.dataframe(
                    proj_df, 
                    width="stretch", 
                    hide_index=True,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(format="MMM YYYY"),
                        "Projected Value": st.column_config.NumberColumn(format="$%.2f")
                    },
                    height=300
                )
    else:
        st.write("No history data available yet.")

# === AUTO REFRESH LOOP ===
if auto_refresh:
    time.sleep(60)
    st.rerun()