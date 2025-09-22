# 📈 Enhanced Stock Analyzer with Interactive Charts

**Made by Neonite - Production Version**

Professional stock analysis tool with AI-powered predictions and interactive charts. Educational use only.

## ✨ Features

- 📊 **Interactive Charts** - Price, volume, technical indicators, and predictions
- 🤖 **ML Predictions** - Ensemble models with confidence scoring
- 📈 **Technical Analysis** - 20+ indicators (RSI, MACD, Bollinger Bands, etc.)
- 🎯 **Market Timing** - Quantified signals and trend analysis
- 🖥️ **Professional GUI** - Desktop interface with real-time charts
- 💻 **Command Line** - Full-featured terminal interface
- 🌐 **Web Dashboard** - Browser-based analysis (optional)

## 🚀 Quick Start

### Option 1: One-Click Setup (Recommended)
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```
*Automatically installs dependencies and launches GUI*

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Choose interface:
python gui.py              # GUI with charts (recommended)
python main.py              # Command line interface
streamlit run dashboard.py  # Web dashboard
```

## 📊 What You Get

### Interactive Charts
- **Price & Volume** - Historical prices with moving averages and Bollinger Bands
- **Technical Indicators** - RSI, MACD, Stochastic with visual zones
- **Price Prediction** - Future forecasts with confidence bands
- **Performance Analysis** - Returns, volatility, and risk metrics

### Analysis Features
- Real-time stock data from Yahoo Finance
- ML ensemble predictions (5 models)
- 20+ technical indicators
- Market timing score (0-100)
- Risk management suggestions
- Feature importance analysis

## 🎯 Usage Examples

### Analyze Apple Stock
```bash
python gui.py
# Enter: AAPL
# Select: 1y period
# Click: Analyze Stock
```

### Popular Stocks to Try
- **AAPL** (Apple), **GOOGL** (Google), **MSFT** (Microsoft)
- **TSLA** (Tesla), **AMZN** (Amazon), **NVDA** (NVIDIA)

## 📋 Requirements

- **Python 3.8+** with tkinter support
- **Internet connection** for stock data
- **Display** for GUI (use CLI on headless systems)

### Dependencies
- pandas, numpy, yfinance, scikit-learn
- matplotlib (for charts), streamlit (for web dashboard)

## 📁 What's Included

**Applications:**
- `gui.py` - Desktop GUI with interactive charts (main application)
- `main.py` - Command line interface
- `dashboard.py` - Web dashboard (Streamlit)
- `performance_analyzer.py` - Backtesting and model validation

**Core Engine:**
- `src/stock_data_fetcher.py` - Yahoo Finance data fetching
- `src/technical_analyzer.py` - Technical indicators & signals  
- `src/stock_predictor.py` - ML ensemble predictions
- `src/finance_advisor.py` - Analysis & recommendations

**Setup & Documentation:**
- `run.sh` / `run.bat` - One-click installation scripts
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## 🛠️ Troubleshooting

**GUI won't start:**
```bash
# Linux: Install tkinter
sudo apt install python3-tk
```

**Module errors:**
```bash
# Use one-click setup
./run.sh  # or run.bat on Windows

# Or manual install
pip install -r requirements.txt
```

**Invalid ticker:**
- Use correct symbols (AAPL, not Apple)
- Check spelling on Yahoo Finance first
- Try popular stocks: AAPL, GOOGL, MSFT, TSLA

## ⚠️ Important Disclaimer

**Educational analysis only - Not financial advice**

- For learning technical analysis and market concepts
- Always consult licensed financial advisors
- Past performance doesn't guarantee future results
- Do your own research before investing

## 🎓 Learning Features

- **Visual Pattern Recognition** - See trends in charts
- **Technical Analysis** - Understand indicator meanings
- **ML Insights** - Feature importance and model confidence
- **Risk Assessment** - Volatility and drawdown analysis

---