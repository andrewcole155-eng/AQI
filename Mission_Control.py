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

@st.cache_data(ttl=300) # History changes slowly, cache longer
def get_portfolio_history(_api):
    try:
        history = _api.get_portfolio_history(period='1M', timeframe='1D')
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    except:
        return pd.DataFrame()

def parse_latest_run_logic(logs):
    signals = {}
    last_run_timestamp = None
    last_run_str = "Unknown"
    
    # Regex to find timestamp YYYY-MM-DD HH:MM:SS
    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})')

    for line in reversed(logs):
        # 1. Parse Logic
        ticker_match = re.search(r'\[([A-Z]+)\]', line)
        if ticker_match:
            ticker = ticker_match.group(1)
            if ticker not in signals:
                clean_msg = line.split(f"[{ticker}]")[-1].strip()
                if "FINAL SIGNAL" in line:
                    signals[ticker] = "✅ " + clean_msg
                elif "Forcing HOLD" in line or "Margin" in line:
                    signals[ticker] = "⏸️ " + clean_msg
                elif "Prediction" in line:
                    signals[ticker] = "🤔 " + clean_msg
                elif "Error" in line:
                    signals[ticker] = "❌ " + clean_msg
                else:
                    signals[ticker] = "ℹ️ " + clean_msg

        # 2. Find Last Timestamp
        if last_run_str == "Unknown":
            match = ts_pattern.search(line)
            if match:
                last_run_str = match.group(1)
                try:
                    last_run_timestamp = datetime.strptime(last_run_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass

    return last_run_str, last_run_timestamp, signals

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
    
    # (Rest of the log logic remains the same...)
    logs = read_bot_logs()
    last_run_str, last_run_dt, parsed_signals = parse_latest_run_logic(logs)
    
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
    c1, c2 = st.columns([3, 4])
    
    with c1:
        st.subheader("Latest Brain Activity")
        if parsed_signals:
            sig_df = pd.DataFrame(list(parsed_signals.items()), columns=["Ticker", "Decision"])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals parsed from recent logs.")

    with c2:
        st.subheader("Active Portfolio")
        if positions:
            pos_data = []
            for p in positions:
                # UPDATE: Access Dictionary keys ['key']
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
            
            # (Rest of the dataframe display logic remains the same...)
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
        # --- PREPARE DATA ---
        dd_df = calculate_drawdown(hist_df)
        ret_df = calculate_daily_returns(hist_df)
        
        # --- ROW 1: EQUITY & DRAWDOWN ---
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.markdown("### 📈 Net Liquidity (30D)")
            fig_eq = px.area(hist_df, x='timestamp', y='equity')
            fig_eq.update_traces(line_color='#00ff41', fillcolor='rgba(0, 255, 65, 0.1)')
            fig_eq.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title=None, showlegend=False,
                height=300,
                # Dynamic scaling to make the chart look "full"
                yaxis=dict(range=[min(hist_df['equity']) * 0.99, max(hist_df['equity']) * 1.01])
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_perf2:
            st.markdown("### 📉 Drawdown (Risk)")
            fig_dd = px.area(dd_df, x='timestamp', y='drawdown')
            fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
            fig_dd.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title=None, showlegend=False,
                height=300,
                yaxis=dict(tickformat=".1%") # Format as percentage
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # --- ROW 2: VOLATILITY & ALLOCATION ---
        st.divider()
        col_perf3, col_perf4 = st.columns(2)

        with col_perf3:
            st.markdown("### 📊 Daily Returns (%)")
            fig_ret = px.bar(ret_df, x='timestamp', y='daily_return')
            fig_ret.update_traces(marker_color=ret_df['color'])
            fig_ret.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title=None, showlegend=False,
                height=300
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        with col_perf4:
            st.markdown("### 🍰 Asset Allocation")
            if positions:
                # Prepare allocation data
                alloc_data = []
                total_market_val = 0
                for p in positions:
                    mv = float(p['qty']) * float(p['current_price'])
                    total_market_val += mv
                    alloc_data.append({"Ticker": p['symbol'], "Value": mv})
                
                # Add Cash to the pie
                cash = float(account['cash'])
                alloc_data.append({"Ticker": "CASH", "Value": cash})
                
                df_alloc = pd.DataFrame(alloc_data)
                
                fig_pie = px.pie(df_alloc, values='Value', names='Ticker', hole=0.4)
                fig_pie.update_traces(textinfo='percent+label')
                fig_pie.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=True,
                    height=300,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.1)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No positions held. Portfolio is 100% Cash.")

    else:
        st.write("No history data available yet.")

# === AUTO REFRESH LOOP ===
if auto_refresh:
    time.sleep(60)
    st.rerun()