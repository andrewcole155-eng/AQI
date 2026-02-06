import streamlit as st
import pandas as pd
import json
import time
import re
import plotly.express as px
import alpaca_trade_api as tradeapi
import gspread
from datetime import datetime

st.set_page_config(
    page_title="Angel V6 Cloud Control",
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
        height: 500px;
        overflow-y: auto;
        font-size: 12px;
        white-space: pre-wrap;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        border-top: 3px solid #00ff41;
    }
    </style>
""", unsafe_allow_html=True)

# === CONNECTIONS (CLOUD COMPATIBLE) ===

@st.cache_resource
def init_alpaca():
    """Connects to Alpaca using Streamlit Secrets."""
    try:
        # Access keys securely from Streamlit Cloud Secrets
        api_key = st.secrets["alpaca"]["API_KEY"]
        secret_key = st.secrets["alpaca"]["SECRET_KEY"]
        base_url = st.secrets["alpaca"]["BASE_URL"]
        
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        return api
    except Exception as e:
        st.error(f"Alpaca Connection Error: {e}")
        return None

def read_bot_logs():
    """Reads logs from Google Sheets (The Bridge)."""
    try:
        # Authenticate using secrets
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        
        # Open the Sheet
        sh = gc.open("Angel_Bot_Logs")
        worksheet = sh.worksheet("logs")
        
        # Get all logs from Column A
        logs = worksheet.col_values(1)
        return logs
    except Exception as e:
        return [f"Google Sheets Error: {e}"]

# === DATA PROCESSING ===
def get_account_data(api):
    try:
        account = api.get_account()
        positions = api.list_positions()
        orders = api.list_orders(status='all', limit=20, direction='desc')
        return account, positions, orders
    except:
        return None, [], []

def get_portfolio_history(api):
    try:
        history = api.get_portfolio_history(period='1M', timeframe='1D')
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    except:
        return pd.DataFrame()

def parse_latest_run_logic(logs):
    signals = {}
    last_run_time = "Unknown"
    
    for line in reversed(logs):
        ticker_match = re.search(r'\[([A-Z]+)\]', line)
        if ticker_match:
            ticker = ticker_match.group(1)
            if ticker not in signals:
                clean_msg = line.split(f"[{ticker}]")[-1].strip()
                if "Signal Generated" in line or "FINAL SIGNAL" in line:
                    signals[ticker] = "✅ " + clean_msg
                elif "Forcing HOLD" in line or "Margin" in line:
                    signals[ticker] = "⏸️ " + clean_msg
                elif "Prediction" in line:
                    signals[ticker] = "🤔 " + clean_msg
                elif "Error" in line:
                    signals[ticker] = "❌ " + clean_msg
                else:
                    signals[ticker] = "ℹ️ " + clean_msg

        if last_run_time == "Unknown" and re.search(r'\d{4}-\d{2}-\d{2}', line):
            match = re.search(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2})', line)
            if match: last_run_time = match.group(1)

    return last_run_time, signals

# === DASHBOARD UI ===
api = init_alpaca()
if not api: st.stop()

if 'last_run' not in st.session_state:
    st.session_state.last_run = time.time()

if st.button("Refresh Data"): st.rerun()

# 1. ACCOUNT
account, positions, orders = get_account_data(api)
if account:
    st.markdown("### 🦅 Angel V6 Cloud Overview")
    col1, col2, col3, col4 = st.columns(4)
    equity = float(account.equity)
    daily_pl_pct = (float(account.equity) - float(account.last_equity)) / float(account.last_equity) * 100
    
    col1.metric("Net Liquidity", f"${equity:,.2f}", f"{daily_pl_pct:.2f}%")
    col2.metric("Day P/L", f"${float(account.equity) - float(account.last_equity):,.2f}")
    col3.metric("Buying Power", f"${float(account.buying_power):,.2f}")
    
    logs = read_bot_logs()
    last_run_time, parsed_signals = parse_latest_run_logic(logs)
    col4.metric("Last Bot Run", last_run_time)

# 2. LOGIC
st.divider()
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown("### 🧠 Latest Decision Logic")
    if parsed_signals:
        sig_df = pd.DataFrame(list(parsed_signals.items()), columns=["Ticker", "Bot Conclusion"])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)
    else:
        st.info("No signals found in logs.")

    st.markdown("### 💼 Active Positions")
    if positions:
        pos_data = [{"Ticker": p.symbol, "Side": p.side.upper(), "P/L ($)": float(p.unrealized_pl), "P/L (%)": float(p.unrealized_plpc)*100} for p in positions]
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.caption("No active positions.")

with c2:
    st.markdown("### 📜 Live Logs (from Google Sheets)")
    st.markdown(f'<div class="terminal-box">{"".join(logs)}</div>', unsafe_allow_html=True)

# 3. CHART
st.divider()
st.markdown("### 📈 Equity Curve")
hist_df = get_portfolio_history(api)
if not hist_df.empty:
    fig = px.area(hist_df, x='timestamp', y='equity', height=300)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        xaxis_title=None, 
        yaxis_title=None, 
        showlegend=False,
        yaxis=dict(range=[3700, max(hist_df['equity']) * 1.02])
    )
    st.plotly_chart(fig, use_container_width=True)