"""
app.py
Streamlit dashboard for the AI Portfolio Optimizer.
Run: streamlit run frontend/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="AI Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# ── Header ──
st.title("📈 AI Portfolio Optimizer")
st.markdown("**Combines FinBERT sentiment analysis with Sharpe ratio optimization**")
st.divider()

# ── Sidebar Inputs ──
st.sidebar.header("Portfolio Settings")

default_tickers = "AAPL,MSFT,GOOGL,AMZN"
ticker_input = st.sidebar.text_input(
    "Stock Tickers (comma separated)",
    value=default_tickers,
    help="Enter US stock symbols separated by commas"
)

period = st.sidebar.selectbox(
    "Historical Period",
    options=["3mo", "6mo", "1y", "2y"],
    index=1
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5
) / 100

st.sidebar.divider()
st.sidebar.markdown("**Current Portfolio Weights (optional)**")
st.sidebar.markdown("*Leave blank to see optimal weights only*")

run_button = st.sidebar.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)

# ── Main Area ──
if not run_button:
    st.info("👈 Enter your stock tickers and click **Optimize Portfolio** to start.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 What It Does")
        st.markdown("""
        - Fetches 6-month stock price history
        - Pulls recent financial news per stock
        - Runs **FinBERT** sentiment on news articles
        - Optimizes weights using **Sharpe ratio**
        - Generates **LLM recommendations**
        """)
    with col2:
        st.markdown("### 🧠 AI Stack")
        st.markdown("""
        - **FinBERT** — financial sentiment model
        - **LangChain** — LLM orchestration
        - **FAISS** — vector search on news
        - **GPT-4** — recommendation generation
        - **scipy** — portfolio optimization
        """)
    with col3:
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This tool is for **educational purposes only**.
        It does not constitute financial advice.
        Always consult a qualified financial advisor.
        """)

else:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.error("Please enter at least 2 stock tickers.")
        st.stop()

    with st.spinner("Fetching stock data and news..."):
        try:
            from src.data.stock_fetcher import fetch_stock_data, calculate_returns
            from src.optimization.portfolio import optimize_portfolio, calculate_current_vs_optimal
            from src.optimization.risk import full_risk_report

            prices = fetch_stock_data(tickers, period=period)
            returns = calculate_returns(prices)
            opt_result = optimize_portfolio(returns, risk_free_rate)
            risk_report = full_risk_report(returns, opt_result["weights"])

            # ── Metrics Row ──
            st.subheader("📊 Optimization Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Annual Return", f"{opt_result['expected_annual_return']}%")
            col2.metric("Annual Volatility", f"{opt_result['annual_volatility']}%")
            col3.metric("Sharpe Ratio", f"{opt_result['sharpe_ratio']}")

            st.divider()

            # ── Weights Chart ──
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Optimal Portfolio Weights")
                weights_df = pd.DataFrame(
                    list(opt_result["weights"].items()),
                    columns=["Ticker", "Weight"]
                )
                weights_df["Weight %"] = weights_df["Weight"] * 100
                fig = px.pie(weights_df, values="Weight %", names="Ticker",
                             hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Stock Volatility")
                vol_df = pd.DataFrame(
                    list(risk_report["volatility_by_stock"].items()),
                    columns=["Ticker", "Volatility (%)"]
                )
                fig2 = px.bar(vol_df, x="Ticker", y="Volatility (%)",
                              color="Ticker", color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig2, use_container_width=True)

            # ── Price History ──
            st.subheader("📈 Price History")
            normalized = prices / prices.iloc[0] * 100
            fig3 = px.line(normalized, title="Normalized Price (Base = 100)")
            st.plotly_chart(fig3, use_container_width=True)

            # ── Risk Report ──
            st.subheader("⚠️ Risk Analysis")
            var = risk_report["value_at_risk"]
            dd = risk_report["max_drawdown"]
            col1, col2 = st.columns(2)
            col1.info(f"**Value at Risk (95%):** {var['interpretation']}")
            col2.warning(f"**Max Drawdown:** {dd['interpretation']}")

            # ── Sentiment placeholder ──
            st.divider()
            st.subheader("🧠 AI Sentiment & Recommendations")
            st.warning("⚙️ Sentiment analysis and LLM recommendations require API keys. Set OPENAI_API_KEY and NEWS_API_KEY in your .env file.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure you have installed all requirements: `pip install -r requirements.txt`")

# ── Footer ──
st.divider()
st.markdown(
    "Built by [Parmod](https://github.com/Parmodk2310) · "
    "[GitHub](https://github.com/Parmodk2310/portfolio-optimizer) · "
    "For educational purposes only"
)