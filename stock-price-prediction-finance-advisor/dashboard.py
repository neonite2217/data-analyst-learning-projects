"""
Interactive Streamlit Dashboard for Stock Analysis
Made by Neonite
"""
import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stock_data_fetcher import StockDataFetcher
from technical_analyzer import TechnicalAnalyzer
from stock_predictor import StockPredictor
from finance_advisor import FinanceAdvisor


st.set_page_config(
    page_title="AI Finance Advisor | Made by Neonite",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Price Prediction + AI Finance Advisor")
st.markdown("*Educational analysis tool - Not financial advice | Made by Neonite*")

# Sidebar inputs
st.sidebar.header("Stock Selection")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="GOOGL")
ticker = ticker_input.upper().strip()
period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

analyze_button = st.sidebar.button("Analyze Stock", type="primary")

if analyze_button and ticker:
    # Validate ticker format
    if len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
        st.error(f"'{ticker}' doesn't appear to be a valid ticker symbol. Ticker symbols are usually 1-5 characters.")
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Validate ticker exists
                is_valid, error_msg = StockDataFetcher.validate_ticker_exists(ticker)
                
                if not is_valid:
                    st.error(f"Error: {error_msg}")
                    st.info("Tips:\n- Check spelling of the ticker symbol\n- Try searching on Yahoo Finance first\n- Some stocks may be delisted or unavailable")
                else:
                    # Fetch data
                    fetcher = StockDataFetcher(ticker)
                    df = fetcher.get_historical_data(period=period)
                    stock_info = fetcher.get_stock_info()
                    
                    if df.empty:
                        st.error(f"No historical data available for {ticker}")
                        st.info("This might be a newly listed stock or data is temporarily unavailable.")
                    else:
                # Technical analysis
                analyzer = TechnicalAnalyzer(df)
                technical_data = analyzer.get_all_indicators()
                
                # Prediction
                predictor = StockPredictor()
                prediction = predictor.simple_trend_prediction(df)
                
                # AI Advisor
                advisor = FinanceAdvisor()
                report = advisor.generate_comprehensive_report(
                    ticker, df, stock_info, technical_data, prediction
                )
                
                # Display company info
                # Data quality warning
                if report['data_quality_warning']:
                    st.warning(f"Warning: {report['data_quality_warning']}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${report['current_price']:.2f}")
                with col2:
                    perf = report['performance_analysis']
                    st.metric("Total Return", f"{perf['total_return_pct']:.2f}%")
                with col3:
                    st.metric("Volatility", f"{perf['volatility_pct']:.2f}%")
                with col4:
                    timing = report['market_timing_analysis']
                    st.metric("Timing Score", f"{timing['timing_score']:.0f}/100")
                
                # Price chart with technical indicators
                st.subheader("📊 Price Chart with Technical Indicators")
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.5, 0.25, 0.25],
                    subplot_titles=('Price & Moving Averages', 'RSI', 'MACD')
                )
                
                # Price and MAs
                fig.add_trace(
                    go.Candlestick(
                        x=technical_data.index,
                        open=technical_data['Open'],
                        high=technical_data['High'],
                        low=technical_data['Low'],
                        close=technical_data['Close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=technical_data.index, y=technical_data['SMA_20'],
                              name='SMA 20', line=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=technical_data.index, y=technical_data['SMA_50'],
                              name='SMA 50', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # RSI
                fig.add_trace(
                    go.Scatter(x=technical_data.index, y=technical_data['RSI'],
                              name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(
                    go.Scatter(x=technical_data.index, y=technical_data['MACD'],
                              name='MACD', line=dict(color='blue')),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=technical_data.index, y=technical_data['MACD_Signal'],
                              name='Signal', line=dict(color='red')),
                    row=3, col=1
                )
                
                fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Two column layout for analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔮 Price Prediction")
                    pred = report['price_prediction']
                    st.write(f"**Predicted Price (30 days):** ${pred['predicted_price']:.2f}")
                    st.write(f"**Expected Change:** {pred['predicted_change_pct']:.2f}%")
                    st.write(f"**Trend:** {pred['trend']}")
                    st.write(f"**Confidence:** {pred['confidence']:.1f}%")
                    
                    st.subheader("📋 Key Metrics")
                    metrics = report['key_metrics']
                    st.write(f"**P/E Ratio:** {metrics['pe_ratio']:.2f}" if metrics['pe_ratio'] else "**P/E Ratio:** N/A")
                    st.write(f"**Market Cap:** ${metrics['market_cap']:,.0f}" if metrics['market_cap'] else "**Market Cap:** N/A")
                    st.write(f"**Beta:** {metrics['beta']:.2f}" if metrics['beta'] else "**Beta:** N/A")
                    st.write(f"**Dividend Yield:** {metrics['dividend_yield']:.2f}%" if metrics['dividend_yield'] else "**Dividend Yield:** N/A")
                
                with col2:
                    st.subheader("⏰ Market Timing Analysis")
                    st.write(f"**Score:** {timing['timing_score']:.1f}/100")
                    st.write(f"**Interpretation:** {timing['interpretation']}")
                    st.write("**Technical Factors:**")
                    for factor in timing['factors']:
                        st.write(f"• {factor}")
                
                # Corrective measures
                st.subheader("💡 Corrective Measures & Suggestions")
                for i, suggestion in enumerate(report['corrective_measures'], 1):
                    st.info(f"{i}. {suggestion}")
                
                # Disclaimer
                st.warning("⚠️ " + report['disclaimer'])
                
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                st.info("Please check the ticker symbol and try again.")

else:
    st.info("Enter a stock ticker in the sidebar and click 'Analyze Stock' to begin")
    
    st.markdown("""
    ### Features:
    - 📊 Real-time stock data from Yahoo Finance (Google Finance data)
    - 📈 Technical analysis with multiple indicators
    - 🔮 Price prediction and trend analysis
    - ⏰ Market timing assessment
    - 💡 Corrective measures and suggestions
    
    ### How to use:
    1. Enter a stock ticker symbol (e.g., GOOGL, AAPL, MSFT)
    2. Select the time period for analysis
    3. Click "Analyze Stock"
    4. Review the comprehensive analysis
    
    **Note:** This tool provides educational analysis only, not buy/sell recommendations.
    """)
