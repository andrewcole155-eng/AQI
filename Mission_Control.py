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

st.set_page_config(
    page_title="Angel V6 Mission Control",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STYLING ===
st.markdown("""
    <style>
    .terminal-box {
        background-color: #0e1117;
        color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
        padding: 10px;
        border: 1px solid #333;
        border-radius: 5px;
        height: 600px;
        overflow-y: auto;
        font-size: 12px;
        white-space: pre-wrap;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00ff41;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR CONFIG ===
with st.sidebar:
    st.header("🦅 Angel Control")
    auto_refresh = st.toggle("Enable Auto-Refresh (60s)", value=True)
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Force Refresh Now", type="primary"):
        st.rerun()

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
        logs = worksheet.col_values(1) # Get all values from Column A
        return logs
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

@st.cache_data(ttl=300)
def get_portfolio_history(_api):
    try:
        # UPDATE: Changed '1M' to 'all' to fetch entire account history
        history = _api.get_portfolio_history(period='all', timeframe='1D')
        
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    except Exception as e:
        # st.error(f"History Error: {e}") # Uncomment for debugging
        return pd.DataFrame()

def parse_latest_run_logic(logs):
    signals = {}
    watchlist = [] # New list for high-potential tickers
    last_run_timestamp = None
    last_run_str = "Unknown"
    
    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})')
    # New regex to find confidence: "Conf: 28.50%"
    conf_pattern = re.compile(r'Conf:\s*([\d\.]+)%?')

    for line in reversed(logs):
        ticker_match = re.search(r'\[([A-Z]+)\]', line)
        if ticker_match:
            ticker = ticker_match.group(1)
            
            # extract confidence if present
            conf_match = conf_pattern.search(line)
            confidence = float(conf_match.group(1)) if conf_match else 0.0

            if ticker not in signals:
                clean_msg = line.split(f"[{ticker}]")[-1].strip()
                if "FINAL SIGNAL" in line:
                    signals[ticker] = "✅ " + clean_msg
                elif "Forcing HOLD" in line or "Margin" in line:
                    signals[ticker] = "⏸️ " + clean_msg
                    # Add to watchlist if it was a near miss (high confidence but held)
                    if confidence > 0.20: 
                        watchlist.append({"Ticker": ticker, "Conf": f"{confidence:.1%}", "Status": "Wait"})
                elif "Prediction" in line:
                    signals[ticker] = "🤔 " + clean_msg
                elif "Error" in line:
                    signals[ticker] = "❌ " + clean_msg
                else:
                    signals[ticker] = "ℹ️ " + clean_msg
                    # Capture raw proposals for watchlist
                    if "RAW PROPOSAL" in line and confidence > 0.20:
                         watchlist.append({"Ticker": ticker, "Conf": f"{confidence:.1%}", "Status": "Watching"})

        if last_run_str == "Unknown":
            match = ts_pattern.search(line)
            if match:
                last_run_str = match.group(1)
                try:
                    last_run_timestamp = datetime.strptime(last_run_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass

    # Deduplicate watchlist (keep latest)
    unique_watchlist = {v['Ticker']:v for v in watchlist}.values()
    return last_run_str, last_run_timestamp, signals, list(unique_watchlist)

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

def calculate_advanced_metrics(hist_df):
    """Calculates institutional metrics from the equity curve."""
    if hist_df.empty: return {}
    
    # Prepare data
    df = hist_df.copy()
    df['daily_return'] = df['equity'].pct_change()
    df['daily_return_pct'] = df['daily_return'] * 100
    
    # 1. CAGR (Annualized Return) - Extrapolated from available data
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    if days < 1: days = 1
    total_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]
    cagr = ((1 + total_return) ** (365 / days)) - 1
    
    # 2. Max Drawdown
    df['peak'] = df['equity'].cummax()
    df['dd'] = (df['equity'] - df['peak']) / df['peak']
    max_dd = df['dd'].min()
    
    # 3. Sharpe Ratio (Assume 0% risk-free rate for simplicity)
    mean_ret = df['daily_return'].mean()
    std_ret = df['daily_return'].std()
    sharpe = (mean_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0
    
    # 4. Sortino Ratio (Downside deviation only)
    downside_std = df[df['daily_return'] < 0]['daily_return'].std()
    sortino = (mean_ret / downside_std) * (252 ** 0.5) if downside_std > 0 else 0
    
    # 5. Daily Win Rate
    wins = len(df[df['daily_return'] > 0])
    total = len(df) - 1 # Minus first day (NaN)
    win_rate = (wins / total) if total > 0 else 0
    
    # 6. MAR Ratio (CAGR / Max Drawdown)
    mar = (cagr / abs(max_dd)) if max_dd != 0 else 0
    
    return {
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Win Rate (Days)": win_rate,
        "MAR Ratio": mar
    }

def create_scorecard_df(metrics):
    """Formats metrics into a DataFrame matching the requested visual style."""
    data = [
        {"METRIC": "CAGR (Est.)", "YOURS": f"{metrics['CAGR']:.1%}", "BENCHMARK": "> 20%", "VERDICT": "🏆 Elite" if metrics['CAGR'] > 0.2 else "😐 Std"},
        {"METRIC": "Sharpe Ratio", "YOURS": f"{metrics['Sharpe Ratio']:.2f}", "BENCHMARK": "> 1.5", "VERDICT": "🔥 Good" if metrics['Sharpe Ratio'] > 1.5 else "😐 Std"},
        {"METRIC": "Max Drawdown", "YOURS": f"{metrics['Max Drawdown']:.1%}", "BENCHMARK": "< 15%", "VERDICT": "🛡️ Safe" if abs(metrics['Max Drawdown']) < 0.15 else "⚠️ High Risk"},
        {"METRIC": "MAR Ratio", "YOURS": f"{metrics['MAR Ratio']:.2f}", "BENCHMARK": "> 1.0", "VERDICT": "🚀 Elite" if metrics['MAR Ratio'] > 1.0 else "😐 Std"},
        {"METRIC": "Sortino Ratio", "YOURS": f"{metrics['Sortino Ratio']:.2f}", "BENCHMARK": "> 2.0", "VERDICT": "💎 Strong" if metrics['Sortino Ratio'] > 2.0 else "😐 Std"},
        {"METRIC": "Daily Win Rate", "YOURS": f"{metrics['Win Rate (Days)']:.0%}", "BENCHMARK": "50-55%", "VERDICT": "✅ Stable" if metrics['Win Rate (Days)'] > 0.5 else "🔻 Low"}
    ]
    return pd.DataFrame(data)




# === DASHBOARD LOGIC ===
api = init_alpaca()
if not api: st.stop()

# 1. ACCOUNT OVERVIEW
account, positions, orders = get_account_data(api)

if account:
    col1, col2, col3, col4 = st.columns(4)
    
    # UPDATE: Access keys as Dictionary ['key'] instead of Object .key
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
    # UPDATED LINE: Unpack the new 4th variable 'watchlist_data'
    last_run_str, last_run_dt, parsed_signals, watchlist_data = parse_latest_run_logic(logs)

    # ... (Keep your existing status logic here) ...
    # Calculate "Time Since Last Run"
    status_label = "Bot Status"
    status_val = "Unknown"
    status_color = "off"
    
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
    # --- 1. MARKET PULSE SECTION ---
    # Calculate simple sentiment from current positions (avg P/L)
    avg_market_move = 0.0
    if positions:
        # Access Dictionary keys ['key']
        avg_market_move = sum([float(p['unrealized_plpc']) for p in positions]) * 100
        # Normalize to 0.0 - 1.0 scale (centered at 0.5)
        sentiment_score = max(0.0, min(1.0, 0.5 + (avg_market_move / 5))) # /5 means +/- 2.5% move hits max
    else:
        sentiment_score = 0.5 # Neutral

    st.markdown("### 🌡️ Market Pulse")
    s_col1, s_col2 = st.columns([5, 1])
    with s_col1:
        st.progress(sentiment_score)
    with s_col2:
        if avg_market_move > 0.5: st.success("BULLISH")
        elif avg_market_move < -0.5: st.error("BEARISH")
        else: st.warning("NEUTRAL")

    st.divider()

    # --- 2. MAIN COLUMNS ---
    c1, c2 = st.columns([3, 4])
    
    with c1:
        st.subheader("🔭 Opportunity Watchlist")
        if watchlist_data:
            wl_df = pd.DataFrame(watchlist_data)
            st.dataframe(wl_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No high-confidence setups detected yet.")

        st.subheader("🧠 Latest Brain Activity")
        if parsed_signals:
            sig_df = pd.DataFrame(list(parsed_signals.items()), columns=["Ticker", "Decision"])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals parsed from recent logs.")

    with c2:
        st.subheader("💼 Active Portfolio")
        if positions:
            pos_data = []
            for p in positions:
                # Access Dictionary keys ['key']
                pl_val = float(p['unrealized_pl'])
                pl_pct = float(p['unrealized_plpc'])
                pos_data.append({
                    "Ticker": p['symbol'], 
                    "Side": p['side'].upper(), 
                    "Qty": float(p['qty']),
                    "Entry": float(p['avg_entry_price']),
                    "P/L ($)": pl_val, 
                    "P/L (%)": pl_pct
                })
            
            df_pos = pd.DataFrame(pos_data)
            
            st.dataframe(
                df_pos,
                use_container_width=True,
                column_config={
                    "P/L ($)": st.column_config.NumberColumn("P/L ($)", format="$%.2f"),
                    "P/L (%)": st.column_config.NumberColumn("P/L (%)", format="%.2f%%"),
                    "Entry": st.column_config.NumberColumn(format="$%.2f"),
                },
                hide_index=True
            )
        else:
            st.caption("No active positions currently held.")

with tab2:
    st.markdown("### Terminal Output (Last 50 Lines)")
    st.markdown(f'<div class="terminal-box">{"".join(logs)}</div>', unsafe_allow_html=True)

with tab3:
    hist_df = get_portfolio_history(api)
    
    if not hist_df.empty:
        # --- CALCULATIONS ---
        metrics = calculate_advanced_metrics(hist_df)
        scorecard_df = create_scorecard_df(metrics)
        dd_df = calculate_drawdown(hist_df)
        
        # --- SECTION 1: THE INSTITUTIONAL SCORECARD ---
        st.markdown("### 🏆 Strategy Scorecard")
        
        # We use a dataframe with column config to mimic the image
        st.dataframe(
            scorecard_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "METRIC": st.column_config.TextColumn("Metric", width="medium"),
                "YOURS": st.column_config.TextColumn("Your Bot", width="small"),
                "BENCHMARK": st.column_config.TextColumn("Target", width="small"),
                "VERDICT": st.column_config.TextColumn("Verdict", width="small"),
            }
        )
        
        st.divider()

        # --- SECTION 2: CHARTS ---
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.markdown("### 📈 Equity Curve")
            fig_eq = px.area(hist_df, x='timestamp', y='equity')
            fig_eq.update_traces(line_color='#00ff41', fillcolor='rgba(0, 255, 65, 0.1)')
            fig_eq.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title=None, showlegend=False,
                height=300,
                yaxis=dict(range=[min(hist_df['equity']) * 0.99, max(hist_df['equity']) * 1.01])
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_perf2:
            st.markdown("### 📉 Risk (Drawdown)")
            fig_dd = px.area(dd_df, x='timestamp', y='drawdown')
            fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
            fig_dd.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title=None, showlegend=False,
                height=300,
                yaxis=dict(tickformat=".1%")
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # --- SECTION 3: WIN RATE HEATMAP (New) ---
        st.markdown("### 📊 Performance by Asset")
        if positions:
            # We can only show CURRENT position performance here without a full trade DB
            # But we can make it look nice
            pos_data = []
            for p in positions:
                pos_data.append({
                    "Ticker": p['symbol'],
                    "Return (%)": float(p['unrealized_plpc']) * 100,
                    "Value": float(p['market_value'])
                })
            df_pos_perf = pd.DataFrame(pos_data)
            
            fig_tree = px.treemap(
                df_pos_perf, 
                path=['Ticker'], 
                values='Value',
                color='Return (%)',
                color_continuous_scale=['#ff4b4b', '#1e1e1e', '#00ff41'],
                color_continuous_midpoint=0
            )
            fig_tree.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_tree, use_container_width=True)
            st.caption("Size = Position Value | Color = Profit/Loss %")
        else:
            st.info("No active positions to analyze.")

    else:
        st.write("No history data available yet.")

# === AUTO REFRESH LOOP ===
if auto_refresh:
    time.sleep(60)
    st.rerun()