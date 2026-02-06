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
        color: #ffb000; /* CHANGED: Soft White for better readability */
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border: 1px solid #333;
        border-radius: 5px;
        height: 600px;
        overflow-y: auto;
        font-size: 13px; /* Slightly larger text */
        line-height: 1.5; /* Better spacing between lines */
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
        # Fetch ALL history first
        history = _api.get_portfolio_history(period='all', timeframe='1D')
        
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # --- UPDATE: Filter for Start Date (24 May 2025) ---
        start_date = pd.Timestamp("2025-05-24")
        df = df[df['timestamp'] >= start_date].copy()
        
        # Sort to ensure calculations are correct
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        # st.error(f"History Error: {e}") 
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
    """Calculates strict Portfolio Metrics (No synthetic Trade Projections)."""
    if hist_df.empty: return {}
    
    df = hist_df.copy()
    df['daily_return'] = df['equity'].pct_change()
    
    # --- 1. RETURN & RISK ---
    days = (df['timestamp'].max() - df['timestamp'].min()).days
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

def calculate_future_projections(hist_df, current_equity):
    """
    Projects equity based on current CAGR for:
    1. End of every month for the next 12 months.
    2. End of the specific current month for the next 10 years.
    """
    if hist_df.empty: return pd.DataFrame()
    
    # 1. Calculate Current CAGR (The Engine)
    df = hist_df.copy()
    days_trading = (df['timestamp'].max() - df['timestamp'].min()).days
    if days_trading < 1: days_trading = 1
    
    # Total return based on official start date vs current live equity
    start_equity = df['equity'].iloc[0]
    total_return = (current_equity - start_equity) / start_equity
    
    # Annualized Rate
    cagr = ((1 + total_return) ** (365 / days_trading)) - 1
    
    # 2. Generate Target Dates
    today = pd.Timestamp.now().normalize()
    target_dates = []
    
    # A. Monthly: End of month for next 12 months (e.g., Feb 26, Mar 26... Feb 27)
    # We use MonthEnd offset to always get the last day
    for i in range(0, 13): 
        future_date = today + pd.tseries.offsets.MonthEnd(i)
        # If we are already past the month end (rare edge case), skip
        if future_date < today: 
            future_date = today + pd.tseries.offsets.MonthEnd(i+1)
        target_dates.append(future_date)
        
    # B. Yearly: End of [Current Month] for next 10 years (e.g., Feb 28, Feb 29...)
    # We start from year 2 because year 1 is covered by the monthly loop above
    current_month_index = today.month 
    for i in range(2, 11): 
        # Calculate future year
        future_year = today.year + i
        # Create timestamp for 1st of that month, then add MonthEnd
        future_dt = pd.Timestamp(year=future_year, month=current_month_index, day=1) + pd.tseries.offsets.MonthEnd(0)
        target_dates.append(future_dt)

    # Deduplicate and Sort
    target_dates = sorted(list(set(target_dates)))
    
    # 3. Calculate Projections
    projections = []
    for date in target_dates:
        # Calculate years from NOW
        years_future = (date - today).days / 365.25
        
        # Future Value Formula: PV * (1+r)^t
        future_val = current_equity * ((1 + cagr) ** years_future)
        
        projections.append({
            "Date": date,
            "Timeline": "Next 12 Months" if years_future <= 1.05 else "10-Year Vision",
            "Projected Value": future_val
        })
        
    return pd.DataFrame(projections), cagr



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
    
    if not hist_df.empty and account:
        current_equity = float(account['equity'])
        
        # --- CALCULATIONS ---
        metrics = calculate_advanced_metrics(hist_df)
        scorecard_df = create_scorecard_df(metrics)
        dd_df = calculate_drawdown(hist_df)
        proj_df, current_cagr = calculate_future_projections(hist_df, current_equity)
        
        # Calculate Institutional Score
        inst_score = calculate_institutional_score(metrics)

        # --- SECTION 1: THE INSTITUTIONAL GAUGE (NEW) ---
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
                        {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.3)'},   # Retail (Red)
                        {'range': [50, 80], 'color': 'rgba(255, 176, 0, 0.3)'}, # Pro (Amber)
                        {'range': [80, 100], 'color': 'rgba(0, 255, 65, 0.3)'}  # Inst (Green)
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': inst_score
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Text Verdict
            if inst_score > 80:
                st.markdown("<div style='text-align: center; color: #00ff41; font-weight: bold;'>🚀 INSTITUTIONAL GRADE</div>", unsafe_allow_html=True)
            elif inst_score > 50:
                st.markdown("<div style='text-align: center; color: #ffb000; font-weight: bold;'>⚡ PROFESSIONAL RETAIL</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: center; color: #ff4b4b; font-weight: bold;'>🎲 DEGEN / RETAIL</div>", unsafe_allow_html=True)

        with col_scorecard:
            st.markdown("### 📊 Metrics Breakdown")
            st.dataframe(
                scorecard_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "METRIC": st.column_config.TextColumn("Metric", width="medium"),
                    "YOURS": st.column_config.TextColumn("Your Bot", width="small"),
                    "BENCHMARK": st.column_config.TextColumn("Target", width="small"),
                    "VERDICT": st.column_config.TextColumn("Verdict", width="small"),
                },
                height=280 # Match gauge height
            )

        st.divider()

        # --- SECTION 2: CHARTS ---
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.markdown("### 📈 Equity Curve")
            fig_eq = px.area(hist_df, x='timestamp', y='equity')
            fig_eq.update_traces(line_color='#00ff41', fillcolor='rgba(0, 255, 65, 0.1)')
            
            # UPDATED: Added yaxis range to start at 3700
            fig_eq.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), 
                xaxis_title=None, 
                yaxis_title=None, 
                showlegend=False, 
                height=300,
                yaxis=dict(range=[3700, None]) # <--- THIS IS THE FIX
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_perf2:
            st.markdown("### 📉 Risk (Drawdown)")
            fig_dd = px.area(dd_df, x='timestamp', y='drawdown')
            fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
            fig_dd.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None, showlegend=False, height=300, yaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_dd, use_container_width=True)

        # --- SECTION 3: QUANT LAB (Analysis) ---
        st.divider()
        st.subheader("🔬 Quant Lab Analysis")
        
        # Prepare Data for Quant Lab
        q_df = hist_df.copy()
        q_df['daily_ret_pct'] = q_df['equity'].pct_change() * 100
        q_df['Day'] = q_df['timestamp'].dt.day_name()
        q_df['vol_30'] = q_df['daily_ret_pct'].rolling(30).std() * (252**0.5) # Rolling Vol
        
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("### 🌊 Rolling Risk (Volatility)")
            if not q_df['vol_30'].dropna().empty:
                fig_vol = px.line(q_df, x='timestamp', y='vol_30', labels={'vol_30': 'Annualized Vol (%)'})
                fig_vol.update_traces(line_color='#ffb000') 
                fig_vol.add_hline(y=q_df['vol_30'].mean(), line_dash="dot", annotation_text="Avg Risk")
                fig_vol.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.info("Need 30 days of data for volatility analysis.")

        with col_q2:
            st.markdown("### 📅 Day-of-Week Performance")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            day_perf = q_df.groupby('Day')['daily_ret_pct'].mean().reindex(day_order)
            fig_day = px.bar(x=day_perf.index, y=day_perf.values, color=day_perf.values, color_continuous_scale=['#ff4b4b', '#1e1e1e', '#00ff41'])
            fig_day.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None)
            st.plotly_chart(fig_day, use_container_width=True)

        # --- SECTION 4: FUTURE PROJECTIONS ---
        st.divider()
        st.markdown(f"### 🔮 Future Projections")
        if not proj_df.empty:
            c_p1, c_p2 = st.columns([2, 1])
            with c_p1:
                fig_proj = px.line(proj_df, x='Date', y='Projected Value', markers=True, color='Timeline')
                fig_proj.update_traces(line_width=3)
                fig_proj.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title=None, yaxis_title=None, height=350, legend=dict(orientation="h", y=1.1, x=0))
                st.plotly_chart(fig_proj, use_container_width=True)
            with c_p2:
                st.dataframe(proj_df, use_container_width=True, hide_index=True)

    else:
        st.write("No history data available yet.")

# === AUTO REFRESH LOOP ===
if auto_refresh:
    time.sleep(60)
    st.rerun()