"""
Enhanced GUI for Stock Price Prediction + AI Finance Advisor
Clean, modern interface with interactive charts using tkinter + matplotlib
Made by Neonite - Version 2.0 with Charts
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys
from pathlib import Path
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stock_data_fetcher import StockDataFetcher
from technical_analyzer import TechnicalAnalyzer
from stock_predictor import StockPredictor
from finance_advisor import FinanceAdvisor


class StockAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Analyzer with Charts - AI Finance Advisor | Made by Neonite")
        self.root.geometry("1400x900")  # Larger window for charts
        self.root.configure(bg='#f5f5f5')
        
        # Data storage for charts
        self.current_data = None
        self.current_report = None
        self.current_technical = None
        self.current_prediction = None
        
        # Style configuration
        self.setup_styles()
        
        # Create main container with paned window for resizable sections
        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls and results
        left_frame = tk.Frame(self.paned_window, bg='#f5f5f5')
        self.paned_window.add(left_frame, weight=1)
        
        # Right panel for charts
        right_frame = tk.Frame(self.paned_window, bg='#f5f5f5')
        self.paned_window.add(right_frame, weight=2)
        
        # Setup left panel
        self.setup_left_panel(left_frame)
        
        # Setup right panel (charts)
        self.setup_charts_panel(right_frame)
    
    def setup_left_panel(self, parent):
        """Setup the left panel with controls and results"""
        left_main = tk.Frame(parent, bg='#f5f5f5')
        left_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.create_header(left_main)
        
        # Input section
        self.create_input_section(left_main)
        
        # Results section (smaller for charts)
        self.create_results_section(left_main)
        
        # Footer
        self.create_footer(left_main)
    
    def setup_charts_panel(self, parent):
        """Setup the right panel with interactive charts"""
        charts_main = tk.Frame(parent, bg='#f5f5f5')
        charts_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Charts header
        charts_header = tk.Label(charts_main,
                                text="📈 Interactive Stock Charts",
                                font=('Segoe UI', 16, 'bold'),
                                bg='#f5f5f5',
                                fg='#333')
        charts_header.pack(pady=(0, 10))
        
        # Chart tabs
        self.chart_notebook = ttk.Notebook(charts_main)
        self.chart_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create chart frames
        self.create_chart_tabs()
        
    def create_chart_tabs(self):
        """Create tabbed interface for different charts"""
        # Price Chart Tab
        self.price_frame = tk.Frame(self.chart_notebook, bg='white')
        self.chart_notebook.add(self.price_frame, text="📊 Price & Volume")
        
        # Technical Indicators Tab
        self.technical_frame = tk.Frame(self.chart_notebook, bg='white')
        self.chart_notebook.add(self.technical_frame, text="📈 Technical Indicators")
        
        # Prediction Chart Tab
        self.prediction_frame = tk.Frame(self.chart_notebook, bg='white')
        self.chart_notebook.add(self.prediction_frame, text="🔮 Price Prediction")
        
        # Performance Tab
        self.performance_frame = tk.Frame(self.chart_notebook, bg='white')
        self.chart_notebook.add(self.performance_frame, text="📊 Performance Analysis")
        
        # Initialize empty charts
        self.init_empty_charts()
    
    def init_empty_charts(self):
        """Initialize empty chart placeholders"""
        for frame, title in [(self.price_frame, "Stock Price & Volume"),
                            (self.technical_frame, "Technical Indicators"),
                            (self.prediction_frame, "Price Prediction"),
                            (self.performance_frame, "Performance Analysis")]:
            
            fig = Figure(figsize=(10, 6), dpi=80, facecolor='white')
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'{title}\n\nAnalyze a stock to see charts here',
                   ha='center', va='center', fontsize=14, color='gray',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_styles(self):
        """Configure modern styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Button style
        style.configure('Accent.TButton',
                       background='#4CAF50',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))
        style.map('Accent.TButton',
                 background=[('active', '#45a049')])
        
        # Combobox style
        style.configure('TCombobox', padding=5)
        
    def create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg='#f5f5f5')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = tk.Label(header_frame,
                        text="Stock Analyzer",
                        font=('Segoe UI', 24, 'bold'),
                        bg='#f5f5f5',
                        fg='#333')
        title.pack()
        
        subtitle = tk.Label(header_frame,
                           text="AI-Powered Finance Advisor",
                           font=('Segoe UI', 11),
                           bg='#f5f5f5',
                           fg='#666')
        subtitle.pack()
        
        credit = tk.Label(header_frame,
                         text="Made by Neonite",
                         font=('Segoe UI', 9, 'italic'),
                         bg='#f5f5f5',
                         fg='#999')
        credit.pack()
        
    def create_input_section(self, parent):
        """Create input controls"""
        input_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, bd=1)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Inner padding
        inner_frame = tk.Frame(input_frame, bg='white')
        inner_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Ticker input
        ticker_frame = tk.Frame(inner_frame, bg='white')
        ticker_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(ticker_frame,
                text="Stock Ticker",
                font=('Segoe UI', 9),
                bg='white',
                fg='#666').pack(anchor=tk.W)
        
        self.ticker_entry = tk.Entry(ticker_frame,
                                     font=('Segoe UI', 11),
                                     width=15,
                                     relief=tk.SOLID,
                                     bd=1)
        self.ticker_entry.pack(pady=(5, 0))
        self.ticker_entry.insert(0, "GOOGL")
        
        # Period selection
        period_frame = tk.Frame(inner_frame, bg='white')
        period_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(period_frame,
                text="Time Period",
                font=('Segoe UI', 9),
                bg='white',
                fg='#666').pack(anchor=tk.W)
        
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(period_frame,
                                    textvariable=self.period_var,
                                    values=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                    state="readonly",
                                    width=12,
                                    font=('Segoe UI', 10))
        period_combo.pack(pady=(5, 0))
        
        # Analyze button
        button_frame = tk.Frame(inner_frame, bg='white')
        button_frame.pack(side=tk.LEFT, padx=(0, 0))
        
        tk.Label(button_frame,
                text=" ",
                font=('Segoe UI', 9),
                bg='white').pack()
        
        self.analyze_btn = ttk.Button(button_frame,
                                      text="Analyze Stock",
                                      style='Accent.TButton',
                                      command=self.analyze_stock)
        self.analyze_btn.pack(pady=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(inner_frame,
                                       mode='indeterminate',
                                       length=200)
        
    def create_results_section(self, parent):
        """Create results display area"""
        results_frame = tk.Frame(parent, bg='white', relief=tk.FLAT, bd=1)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(results_frame,
                                                      wrap=tk.WORD,
                                                      font=('Consolas', 10),
                                                      bg='white',
                                                      fg='#333',
                                                      relief=tk.FLAT,
                                                      padx=20,
                                                      pady=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for styling
        self.results_text.tag_configure('header', font=('Segoe UI', 14, 'bold'), foreground='#2196F3')
        self.results_text.tag_configure('subheader', font=('Segoe UI', 11, 'bold'), foreground='#333')
        self.results_text.tag_configure('metric', font=('Consolas', 10, 'bold'), foreground='#4CAF50')
        self.results_text.tag_configure('warning', font=('Consolas', 10), foreground='#FF9800')
        self.results_text.tag_configure('info', font=('Consolas', 10), foreground='#666')
        
        # Initial message
        self.show_welcome_message()
        
    def create_footer(self, parent):
        """Create footer with disclaimer"""
        footer_frame = tk.Frame(parent, bg='#f5f5f5')
        footer_frame.pack(fill=tk.X, pady=(15, 0))
        
        disclaimer = tk.Label(footer_frame,
                             text="Educational analysis only - Not financial advice",
                             font=('Segoe UI', 9),
                             bg='#f5f5f5',
                             fg='#999')
        disclaimer.pack()
        
    def show_welcome_message(self):
        """Display welcome message"""
        self.results_text.delete(1.0, tk.END)
        
        welcome = """
        Welcome to Stock Analyzer!
        
        Get started:
        1. Enter a stock ticker (e.g., GOOGL, AAPL, MSFT)
        2. Select a time period
        3. Click "Analyze Stock"
        
        You'll receive:
        - Performance analysis
        - Technical indicators
        - Price predictions
        - Market timing insights
        - Risk management suggestions
        
        Try these popular stocks:
        GOOGL (Google), AAPL (Apple), MSFT (Microsoft),
        TSLA (Tesla), AMZN (Amazon), NVDA (NVIDIA)
        """
        
        self.results_text.insert(tk.END, welcome, 'info')
        
    def analyze_stock(self):
        """Perform stock analysis in background thread"""
        ticker = self.ticker_entry.get().strip().upper()
        
        if not ticker:
            messagebox.showwarning("Input Required", "Please enter a stock ticker symbol")
            return
        
        # Validate ticker format
        if len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
            messagebox.showerror("Invalid Ticker", 
                               f"'{ticker}' doesn't appear to be a valid ticker symbol.\n\n"
                               "Ticker symbols are usually 1-5 characters (e.g., AAPL, GOOGL, MSFT)")
            return
        
        # Disable button and show progress
        self.analyze_btn.config(state='disabled')
        self.progress.pack(side=tk.LEFT, padx=(15, 0), pady=(5, 0))
        self.progress.start(10)
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Analyzing {ticker}...\n\n", 'info')
        
        # Run analysis in background thread
        thread = Thread(target=self._run_analysis, args=(ticker,))
        thread.daemon = True
        thread.start()
        
    def _run_analysis(self, ticker):
        """Background analysis task"""
        try:
            period = self.period_var.get()
            
            # Validate ticker first
            is_valid, error_msg = StockDataFetcher.validate_ticker_exists(ticker)
            if not is_valid:
                self.root.after(0, self._show_error, 
                              f"{error_msg}\n\nTips:\n"
                              "• Check spelling of the ticker symbol\n"
                              "• Try searching on Yahoo Finance first\n"
                              "• Some stocks may be delisted or unavailable")
                return
            
            # Fetch data
            fetcher = StockDataFetcher(ticker)
            df = fetcher.get_historical_data(period=period)
            stock_info = fetcher.get_stock_info()
            
            if df.empty:
                self.root.after(0, self._show_error, 
                              f"No historical data available for {ticker}\n\n"
                              "This might be a newly listed stock or data is temporarily unavailable.")
                return
            
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
            
            # Store data for charts
            self.current_data = df
            self.current_report = report
            self.current_technical = technical_data
            self.current_prediction = prediction
            
            # Display results and update charts
            self.root.after(0, self._display_results, report)
            self.root.after(0, self._update_charts)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
        finally:
            self.root.after(0, self._analysis_complete)
            
    def _display_results(self, report):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, f"\n{report['company_name']}\n", 'header')
        self.results_text.insert(tk.END, f"{report['symbol']} • {report['analysis_date']}\n", 'info')
        self.results_text.insert(tk.END, f"Data: {report['data_days']} days\n\n", 'info')
        
        # Data quality warning
        if report['data_quality_warning']:
            self.results_text.insert(tk.END, f"Warning: {report['data_quality_warning']}\n\n", 'warning')
        
        # Current Price
        self.results_text.insert(tk.END, f"Current Price: ${report['current_price']:.2f}\n\n", 'metric')
        
        # Performance
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        self.results_text.insert(tk.END, "📊 PERFORMANCE ANALYSIS\n", 'subheader')
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        
        perf = report['performance_analysis']
        self.results_text.insert(tk.END, f"Total Return:        {perf['total_return_pct']:>8.2f}%\n", 'info')
        self.results_text.insert(tk.END, f"Volatility:          {perf['volatility_pct']:>8.2f}%\n", 'info')
        self.results_text.insert(tk.END, f"Current Trend:       {perf['current_trend']:>12}\n", 'info')
        self.results_text.insert(tk.END, f"52-Week High:        ${perf['52w_high']:>10.2f}\n", 'info')
        self.results_text.insert(tk.END, f"52-Week Low:         ${perf['52w_low']:>10.2f}\n", 'info')
        self.results_text.insert(tk.END, f"Position in Range:   {perf['price_position_in_52w_range']:>8.1f}%\n\n", 'info')
        
        # Key Metrics
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        self.results_text.insert(tk.END, "📋 KEY METRICS\n", 'subheader')
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        
        metrics = report['key_metrics']
        pe = metrics['pe_ratio']
        self.results_text.insert(tk.END, f"P/E Ratio:           {pe:>10.2f}\n" if pe else "P/E Ratio:                 N/A\n", 'info')
        
        mcap = metrics['market_cap']
        self.results_text.insert(tk.END, f"Market Cap:          ${mcap:>12,.0f}\n" if mcap else "Market Cap:                N/A\n", 'info')
        
        beta = metrics['beta']
        self.results_text.insert(tk.END, f"Beta:                {beta:>10.2f}\n" if beta else "Beta:                      N/A\n", 'info')
        
        div = metrics['dividend_yield']
        self.results_text.insert(tk.END, f"Dividend Yield:      {div:>8.2f}%\n\n" if div else "Dividend Yield:            N/A\n\n", 'info')
        
        # Prediction
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        self.results_text.insert(tk.END, "🔮 PRICE PREDICTION (30 Days)\n", 'subheader')
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        
        pred = report['price_prediction']
        self.results_text.insert(tk.END, f"Predicted Price:     ${pred['predicted_price']:>10.2f}\n", 'metric')
        self.results_text.insert(tk.END, f"Expected Change:     {pred['predicted_change_pct']:>8.2f}%\n", 'info')
        self.results_text.insert(tk.END, f"Trend:               {pred['trend']:>12}\n", 'info')
        self.results_text.insert(tk.END, f"Confidence:          {pred['confidence']:>8.1f}%\n\n", 'info')
        
        # Market Timing
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        self.results_text.insert(tk.END, "⏰ MARKET TIMING ANALYSIS\n", 'subheader')
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        
        timing = report['market_timing_analysis']
        self.results_text.insert(tk.END, f"Timing Score:        {timing['timing_score']:>8.1f}/100\n", 'metric')
        self.results_text.insert(tk.END, f"\n{timing['interpretation']}\n\n", 'info')
        
        self.results_text.insert(tk.END, "Technical Factors:\n", 'info')
        for i, factor in enumerate(timing['factors'], 1):
            self.results_text.insert(tk.END, f"  {i}. {factor}\n", 'info')
        
        # Corrective Measures
        self.results_text.insert(tk.END, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        self.results_text.insert(tk.END, "💡 CORRECTIVE MEASURES & SUGGESTIONS\n", 'subheader')
        self.results_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", 'info')
        
        for i, suggestion in enumerate(report['corrective_measures'], 1):
            self.results_text.insert(tk.END, f"{i}. {suggestion}\n", 'warning')
        
        self.results_text.insert(tk.END, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n", 'info')
        
        # Scroll to top
        self.results_text.see(1.0)
        
    def _show_error(self, error_msg):
        """Display error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"\nError: {error_msg}\n\n", 'warning')
        self.results_text.insert(tk.END, "Please check the ticker symbol and try again.\n", 'info')
        
    def _analysis_complete(self):
        """Re-enable controls after analysis"""
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_btn.config(state='normal')
    
    def _update_charts(self):
        """Update all charts with current data"""
        if self.current_data is None:
            return
        
        try:
            self._create_price_chart()
            self._create_technical_chart()
            self._create_prediction_chart()
            self._create_performance_chart()
        except Exception as e:
            print(f"Chart update error: {e}")
    
    def _create_price_chart(self):
        """Create price and volume chart"""
        # Clear existing chart
        for widget in self.price_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(12, 8), dpi=80, facecolor='white')
        
        # Price subplot
        ax1 = fig.add_subplot(211)
        ax1.plot(self.current_data.index, self.current_data['Close'], 
                linewidth=2, color='#2196F3', label='Close Price')
        ax1.plot(self.current_data.index, self.current_technical['SMA_20'], 
                linewidth=1, color='#FF9800', alpha=0.8, label='SMA 20')
        ax1.plot(self.current_data.index, self.current_technical['SMA_50'], 
                linewidth=1, color='#4CAF50', alpha=0.8, label='SMA 50')
        
        # Bollinger Bands
        ax1.fill_between(self.current_data.index, 
                        self.current_technical['BB_Upper'], 
                        self.current_technical['BB_Lower'],
                        alpha=0.1, color='gray', label='Bollinger Bands')
        
        ax1.set_title(f"{self.current_report['symbol']} - Stock Price & Moving Averages", 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume subplot
        if 'Volume' in self.current_data.columns:
            ax2 = fig.add_subplot(212)
            ax2.bar(self.current_data.index, self.current_data['Volume'], 
                   alpha=0.6, color='#9C27B0', width=0.8)
            ax2.set_title('Trading Volume', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.price_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _create_technical_chart(self):
        """Create technical indicators chart"""
        # Clear existing chart
        for widget in self.technical_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(12, 10), dpi=80, facecolor='white')
        
        # RSI subplot
        ax1 = fig.add_subplot(311)
        ax1.plot(self.current_data.index, self.current_technical['RSI'], 
                linewidth=2, color='#FF5722', label='RSI')
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax1.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax1.fill_between(self.current_data.index, 30, 70, alpha=0.1, color='gray')
        ax1.set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('RSI', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # MACD subplot
        ax2 = fig.add_subplot(312)
        ax2.plot(self.current_data.index, self.current_technical['MACD'], 
                linewidth=2, color='#2196F3', label='MACD')
        ax2.plot(self.current_data.index, self.current_technical['MACD_Signal'], 
                linewidth=2, color='#FF9800', label='Signal')
        ax2.bar(self.current_data.index, self.current_technical['MACD_Histogram'], 
               alpha=0.6, color='gray', width=0.8, label='Histogram')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('MACD (Moving Average Convergence Divergence)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MACD', fontsize=10)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Stochastic subplot
        ax3 = fig.add_subplot(313)
        if 'Stoch_K' in self.current_technical.columns:
            ax3.plot(self.current_data.index, self.current_technical['Stoch_K'], 
                    linewidth=2, color='#4CAF50', label='%K')
            ax3.plot(self.current_data.index, self.current_technical['Stoch_D'], 
                    linewidth=2, color='#FF5722', label='%D')
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
            ax3.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
            ax3.fill_between(self.current_data.index, 20, 80, alpha=0.1, color='gray')
            ax3.set_title('Stochastic Oscillator', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Stochastic', fontsize=10)
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.technical_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _create_prediction_chart(self):
        """Create price prediction chart"""
        # Clear existing chart
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(12, 8), dpi=80, facecolor='white')
        ax = fig.add_subplot(111)
        
        # Historical prices (last 60 days)
        recent_data = self.current_data.tail(60)
        ax.plot(recent_data.index, recent_data['Close'], 
               linewidth=2, color='#2196F3', label='Historical Price')
        
        # Current price point
        current_price = self.current_prediction['current_price']
        current_date = self.current_data.index[-1]
        ax.scatter([current_date], [current_price], 
                  color='red', s=100, zorder=5, label='Current Price')
        
        # Future prediction
        days_ahead = self.current_prediction.get('days_ahead', 30)
        future_dates = pd.date_range(start=current_date + timedelta(days=1), 
                                   periods=days_ahead, freq='D')
        
        # Simple linear projection for visualization
        predicted_price = self.current_prediction['predicted_price']
        
        # Create prediction line
        prediction_line = np.linspace(current_price, predicted_price, days_ahead)
        ax.plot(future_dates, prediction_line, 
               linewidth=3, color='#4CAF50', linestyle='--', 
               label=f'Prediction ({days_ahead} days)')
        
        # Confidence bands
        confidence = self.current_prediction['confidence'] / 100
        volatility = self.current_data['Close'].pct_change().std()
        
        upper_band = prediction_line * (1 + volatility * (1 - confidence))
        lower_band = prediction_line * (1 - volatility * (1 - confidence))
        
        ax.fill_between(future_dates, lower_band, upper_band, 
                       alpha=0.2, color='#4CAF50', label='Confidence Band')
        
        # Prediction point
        ax.scatter([future_dates[-1]], [predicted_price], 
                  color='#4CAF50', s=150, zorder=5, 
                  label=f'Target: ${predicted_price:.2f}')
        
        # Formatting
        ax.set_title(f"{self.current_report['symbol']} - Price Prediction", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add prediction details as text
        change_pct = self.current_prediction['predicted_change_pct']
        confidence_pct = self.current_prediction['confidence']
        trend = self.current_prediction['trend']
        
        info_text = f"Predicted Change: {change_pct:+.2f}%\nConfidence: {confidence_pct:.1f}%\nTrend: {trend}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.prediction_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _create_performance_chart(self):
        """Create performance analysis chart"""
        # Clear existing chart
        for widget in self.performance_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(12, 10), dpi=80, facecolor='white')
        
        # Returns distribution
        ax1 = fig.add_subplot(221)
        returns = self.current_data['Close'].pct_change().dropna()
        ax1.hist(returns, bins=30, alpha=0.7, color='#2196F3', edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns.mean():.4f}')
        ax1.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Daily Return')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax2 = fig.add_subplot(222)
        cumulative_returns = (1 + returns).cumprod()
        ax2.plot(cumulative_returns.index, cumulative_returns, 
                linewidth=2, color='#4CAF50')
        ax2.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True, alpha=0.3)
        
        # Volatility (rolling)
        ax3 = fig.add_subplot(223)
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        ax3.plot(rolling_vol.index, rolling_vol, 
                linewidth=2, color='#FF5722')
        ax3.set_title('Rolling Volatility (20-day)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Annualized Volatility')
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # Calculate metrics
        total_return = self.current_report['performance_analysis']['total_return_pct']
        volatility = self.current_report['performance_analysis']['volatility_pct']
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min()
        
        metrics_text = f"""Performance Metrics:
        
Total Return: {total_return:.2f}%
Annualized Volatility: {volatility:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {max_drawdown:.2f}%

Current Position:
52W Range: {self.current_report['performance_analysis']['price_position_in_52w_range']:.1f}%
Trend: {self.current_report['performance_analysis']['current_trend']}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.performance_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = StockAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
