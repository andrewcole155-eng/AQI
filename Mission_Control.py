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
        orders = [o._raw for o in _api.list_orders(status='all', limit=20, direction='desc')]
        return account, positions, orders
    except:
        return None, [], []

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
    3. Neural Conviction (Latest confidence score for ALL tickers)
    """
    signals = {}
    watchlist = [] 
    neural_conviction = {} # NEW: Stores {Ticker: Confidence}
    last_run_timestamp = None
    last_run_str = "Unknown"
    
    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})')
    conf_pattern = re.compile(r'Conf:\s*([\d\.]+)%?')
    
    # We iterate reversed to get the LATEST log entry for each ticker first
    for line in reversed(logs):
        ticker_match = re.search(r'\[([A-Z]+)\]', line)
        if ticker_match:
            ticker = ticker_match.group(1)
            
            # Extract confidence if present
            conf_match = conf_pattern.search(line)
            confidence = float(conf_match.group(1)) if conf_match else 0.0
            
            # NEW: Update Neural Conviction if not already set (since we are reading reversed)
            if ticker not in neural_conviction and confidence > 0:
                neural_conviction[ticker] = confidence

            # Signal Logic (Same as before)
            if ticker not in signals:
                clean_msg = line.split(f"[{ticker}]")[-1].strip()
                if "FINAL SIGNAL" in line:
                    signals[ticker] = "✅ " + clean_msg
                elif "Forcing HOLD" in line or "Margin" in line:
                    signals[ticker] = "⏸️ " + clean_msg
                    if confidence > 0.20: 
                        watchlist.append({"Ticker": ticker, "Conf": f"{confidence:.1%}", "Status": "Wait"})
                elif "Prediction" in line:
                    signals[ticker] = "🤔 " + clean_msg
                elif "Error" in line:
                    signals[ticker] = "❌ " + clean_msg
                else:
                    # Generic info, but capture raw proposals
                    if "RAW PROPOSAL" in line and confidence > 0.20:
                         watchlist.append({"Ticker": ticker, "Conf": f"{confidence:.1%}", "Status": "Watching"})

        # Timestamp extraction
        if last_run_str == "Unknown":
            match = ts_pattern.search(line)
            if match:
                last_run_str = match.group(1)
                try:
                    last_run_timestamp = datetime.strptime(last_run_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass

    # Deduplicate watchlist
    unique_watchlist = {v['Ticker']:v for v in watchlist}.values()
    
    # NEW: Return 5 items now (added neural_conviction)
    return last_run_str, last_run_timestamp, signals, list(unique_watchlist), neural_conviction

def calculate_drawdown(df):
    """Calculates the Drawdown (percentage drop from peak equity)."""
    df = df.copy()
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
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
    """Calculates strict Portfolio Metrics (No synthetic Trade Projections)."""
    if hist_df.empty: return {}
    
    df = hist_df.copy()
    df['daily_return'] = df['equity'].pct_change()
    
    # --- 1. RETURN & RISK ---
    # UPDATE: Force strict start date for CAGR time calculation
    start_date = pd.Timestamp("2025-05-24")
    current_date = df['timestamp'].max()
    
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

    # --- 2. PROFITABILITY ---
    df['diff'] = df['equity'].diff()
    gross_profit = df[df['diff'] > 0]['diff'].sum()
    gross_loss = abs(df[df['diff'] < 0]['diff'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # Win Rate (Days)
    wins = len(df[df['diff'] > 0])
    total_active = len(df[df['diff'] != 0])
    win_rate = (wins / total_active) if total_active > 0 else 0

    return {
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "MAR Ratio": mar,
        "Profit Factor": profit_factor,
        "Win Rate (Daily)": win_rate
    }

def create_scorecard_df(metrics):
    """Formats the simplified Strategy Scorecard."""
    
    data = [
        # --- RETURN ---
        {"METRIC": "CAGR (Account)", "YOURS": f"{metrics['CAGR']:.1%}", "BENCHMARK": "> 20%", "VERDICT": "🏆 Elite" if metrics['CAGR'] > 0.2 else "😐 Std"},
        {"METRIC": "MAR Ratio", "YOURS": f"{metrics['MAR Ratio']:.2f}", "BENCHMARK": "> 1.0", "VERDICT": "🚀 Elite" if metrics['MAR Ratio'] > 1.0 else "😐 Std"},
        
        # --- RISK ---
        {"METRIC": "Max Drawdown", "YOURS": f"{metrics['Max Drawdown']:.1%}", "BENCHMARK": "< 15%", "VERDICT": "🛡️ Safe" if abs(metrics['Max Drawdown']) < 0.15 else "⚠️ High Risk"},
        {"METRIC": "Sharpe Ratio", "YOURS": f"{metrics['Sharpe Ratio']:.2f}", "BENCHMARK": "> 1.5", "VERDICT": "🔥 Good" if metrics['Sharpe Ratio'] > 1.5 else "😐 Std"},
        {"METRIC": "Sortino Ratio", "YOURS": f"{metrics['Sortino Ratio']:.2f}", "BENCHMARK": "> 2.0", "VERDICT": "💎 Strong" if metrics['Sortino Ratio'] > 2.0 else "😐 Std"},

        # --- CONSISTENCY ---
        {"METRIC": "Profit Factor", "YOURS": f"{metrics['Profit Factor']:.2f}", "BENCHMARK": "> 1.5", "VERDICT": "💰 Rich" if metrics['Profit Factor'] > 1.5 else "😐 Std"},
        {"METRIC": "Daily Win Rate", "YOURS": f"{metrics['Win Rate (Daily)']:.0%}", "BENCHMARK": "50-55%", "VERDICT": "✅ Stable" if metrics['Win Rate (Daily)'] > 0.5 else "🔻 Low"},
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
    col1, col2, col3, col4 = st.columns(4)
    
    equity = float(account['equity'])
    last_equity = float(account['last_equity'])
    buying_power = float(account['buying_power'])
    
    daily_pl_pct = (equity - last_equity) / last_equity * 100
    daily_pl_abs = equity - last_equity
    
    col1.metric("Net Liquidity", f"${equity:,.2f}", f"{daily_pl_pct:.2f}%")
    col2.metric("Day P/L", f"${daily_pl_abs:,.2f}")
    col3.metric("Buying Power", f"${buying_power:,.2f}")
    
    # Process Logs
    logs = read_bot_logs()
    last_run_str, last_run_dt, parsed_signals, watchlist_data, conviction_data = parse_latest_run_logic(logs)

    # Calculate "Time Since Last Run"
    status_label = "Bot Status"
    status_val = "Unknown"
    
    if last_run_dt:
        diff = datetime.now() - last_run_dt 
        minutes_ago = int(diff.total_seconds() / 60)
        
        if minutes_ago < 10:
            status_val = "🟢 Active"
        elif minutes_ago < 60:
            status_val = f"🟡 Idle ({minutes_ago}m)"
        else:
            status_val = f"🔴 Stale ({int(minutes_ago/60)}h)"
    
    col4.metric(status_label, status_val, delta=f"Last Log: {last_run_str}", delta_color="off")

st.divider()

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
    s_col1, s_col2 = st.columns([5, 1])
    with s_col1:
        st.progress(sentiment_score)
    with s_col2:
        if avg_market_move > 0.5: st.success("BULLISH")
        elif avg_market_move < -0.5: st.error("BEARISH")
        else: st.warning("NEUTRAL")

    st.divider()

    # --- 2. NEURAL CONVICTION RADAR ---
    st.subheader("🧠 Neural Conviction Levels")
    if conviction_data:
        df_conv = pd.DataFrame(list(conviction_data.items()), columns=['Ticker', 'Confidence'])
        
        fig_conf = px.bar(
            df_conv, 
            x='Ticker', 
            y='Confidence', 
            color='Confidence',
            color_continuous_scale=['#2d2d2d', '#ffb000', '#00ff41'], 
            range_y=[0, 100],
            text_auto='.1f'
        )
        fig_conf.update_layout(
            height=250, 
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title=None, 
            yaxis_title="Confidence %",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#cccccc'},
            yaxis=dict(showgrid=True, gridcolor='#333'),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Waiting for first model run to populate conviction data...")

    st.divider()

    # --- 3. MAIN COLUMNS ---
    c1, c2 = st.columns([3, 4])
    
    with c1:
        st.subheader("🔭 Opportunity Watchlist")
        if watchlist_data:
            wl_df = pd.DataFrame(watchlist_data)
            # FIX: Updated width parameter
            st.dataframe(wl_df, width="stretch", hide_index=True)
        else:
            st.caption("No high-confidence setups detected yet.")

        st.subheader("📝 Decision Log")
        if parsed_signals:
            sig_df = pd.DataFrame(list(parsed_signals.items()), columns=["Ticker", "Decision"])
            # FIX: Updated width parameter
            st.dataframe(sig_df, width="stretch", hide_index=True)
        else:
            st.info("No signals parsed from recent logs.")

    with c2:
        st.subheader("💼 Active Portfolio")
        if positions:
            pos_data = []
            for p in positions:
                pl_val = float(p['unrealized_pl'])
                pl_pct = float(p['unrealized_plpc']) * 100
                pos_data.append({
                    "Ticker": p['symbol'], 
                    "Side": p['side'].upper(), 
                    "Qty": float(p['qty']),
                    "Entry": float(p['avg_entry_price']),
                    "P/L ($)": pl_val, 
                    "P/L (%)": pl_pct
                })
            
            df_pos = pd.DataFrame(pos_data)
            # FIX: Updated width parameter
            st.dataframe(
                df_pos,
                width="stretch",
                column_config={
                    "P/L ($)": st.column_config.NumberColumn("P/L ($)", format="$%.2f"),
                    "P/L (%)": st.column_config.NumberColumn("P/L (%)", format="%.2f%%"),
                    "Entry": st.column_config.NumberColumn(format="$%.2f"),
                },
                hide_index=True
            )
        else:
            st.caption("No active positions currently held.")

        # --- RECENT ORDERS ---
        st.divider()
        st.subheader("📜 Recent Orders")
        if orders:
            order_data = []
            for o in orders[:5]: 
                t = o['created_at']
                t_fmt = t[5:16].replace('T', ' ') 
                
                order_data.append({
                    "Time": t_fmt,
                    "Ticker": o['symbol'],
                    "Side": o['side'].upper(),
                    "Qty": o['qty'],
                    "Status": o['status'].title()
                })
            
            df_orders = pd.DataFrame(order_data)
            # FIX: Updated width parameter
            st.dataframe(df_orders, width="stretch", hide_index=True)
        else:
            st.caption("No recent orders found.")

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
    # 1. Get History (Now Raw)
    hist_df_raw = get_portfolio_history(api)
    
    if not hist_df_raw.empty and account:
        current_equity_raw = float(account['equity'])

        # Ensure timezone awareness matches
        now_ts = pd.Timestamp.now(tz='UTC') 
        if hist_df_raw['timestamp'].dt.tz is None:
            now_ts = pd.Timestamp.now()

        # Append LIVE Raw Data
        live_row = pd.DataFrame([{
            'timestamp': now_ts, 
            'equity': current_equity_raw
        }])
        hist_df_raw = pd.concat([hist_df_raw, live_row], ignore_index=True)

        # === DATA FORK: Create "Adjusted" Copy for Metrics Only ===
        # We create a separate dataframe that subtracts the deposits.
        # This allows us to calculate HONEST CAGR while showing REAL charts.
        hist_df_adj = hist_df_raw.copy()
        
        # Apply Adjustments to the "Metric" dataframe only
        mask_1 = hist_df_adj['timestamp'] >= pd.Timestamp("2026-01-24", tz=hist_df_adj['timestamp'].dt.tz)
        hist_df_adj.loc[mask_1, 'equity'] = hist_df_adj.loc[mask_1, 'equity'] - 68.10

        mask_2 = hist_df_adj['timestamp'] >= pd.Timestamp("2026-02-12", tz=hist_df_adj['timestamp'].dt.tz)
        hist_df_adj.loc[mask_2, 'equity'] = hist_df_adj.loc[mask_2, 'equity'] - 69.81
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
        
        # FIXED LINE BELOW: Added the closing parenthesis
        proj_df = calculate_future_projections(current_equity_raw, projection_rate)

        # --- SECTION 1: THE INSTITUTIONAL GAUGE ---
        col_gauge, col_scorecard = st.columns([1, 2])
        
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
            fig_dow.update_layout(
                yaxis=dict(title="Avg Return (%)", showgrid=True, gridcolor='#333'),
                yaxis2=dict(title="Win Rate (%)", overlaying='y', side='right', range=[0, 110], showgrid=False),
                showlegend=False,
                height=350,
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
                height=350,
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
                height=600 
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