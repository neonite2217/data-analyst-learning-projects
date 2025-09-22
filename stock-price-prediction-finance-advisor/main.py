import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stock_data_fetcher import StockDataFetcher
from technical_analyzer import TechnicalAnalyzer
from stock_predictor import StockPredictor
from finance_advisor import FinanceAdvisor
import pandas as pd


def analyze_stock(ticker: str, period: str = "1y"):
    """
    Perform comprehensive stock analysis
    
    Args:
        ticker: Stock ticker symbol (e.g., 'GOOGL', 'AAPL')
        period: Time period for analysis (1y, 2y, 5y, etc.)
    """
    # Normalize ticker input
    ticker = ticker.upper().strip()
    
    print(f"\n{'='*80}")
    print(f"Stock Analysis for {ticker}")
    print(f"{'='*80}\n")
    
    # Validate ticker first
    print("Validating ticker symbol...")
    is_valid, error_msg = StockDataFetcher.validate_ticker_exists(ticker)
    
    if not is_valid:
        print(f"Error: {error_msg}")
        print("\nTips:")
        print("  - Check spelling of the ticker symbol")
        print("  - Try searching on Yahoo Finance first")
        print("  - Use the full ticker (e.g., 'GOOGL' not 'GOOG')")
        print("  - Some stocks may be delisted or unavailable")
        return None
    
    print("Ticker validated\n")
    
    # Fetch stock data
    print("Fetching stock data from Yahoo Finance...")
    try:
        fetcher = StockDataFetcher(ticker)
        df = fetcher.get_historical_data(period=period)
        stock_info = fetcher.get_stock_info()
        
        if df.empty:
            print(f"Error: No historical data available for {ticker}")
            print("This might be a newly listed stock or data is temporarily unavailable.")
            return None
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None
    
    print(f"Retrieved {len(df)} days of historical data")
    print(f"Company: {stock_info.get('name', 'N/A')}")
    print(f"Sector: {stock_info.get('sector', 'N/A')}\n")
    
    # Technical analysis
    print("Performing technical analysis...")
    analyzer = TechnicalAnalyzer(df)
    technical_data = analyzer.get_all_indicators()
    print("Technical indicators calculated\n")
    
    # Price prediction
    print("Generating advanced ML price prediction...")
    predictor = StockPredictor()
    
    # Try ensemble prediction first, fallback to simple if needed
    try:
        prediction = predictor.ensemble_prediction(df, days_ahead=7)  # 1 week prediction
        print(f"Ensemble prediction complete using {prediction.get('model_count', 0)} models")
    except Exception as e:
        print(f"Ensemble prediction failed, using simple method: {e}")
        prediction = predictor.simple_trend_prediction(df, days_ahead=30)
    
    print("Prediction complete\n")
    
    # AI Finance Advisor Analysis
    print("AI Finance Advisor Analysis...")
    advisor = FinanceAdvisor()
    report = advisor.generate_comprehensive_report(
        ticker, df, stock_info, technical_data, prediction
    )
    
    # Enhanced analysis - get signal summary and feature importance
    signal_summary = analyzer.get_signal_summary()
    feature_importance = predictor.get_feature_importance()
    
    # Display results
    display_report(report, signal_summary, feature_importance)
    
    return report


def display_report(report: dict, signal_summary: dict = None, feature_importance: dict = None):
    """Display the enhanced analysis report in a readable format"""
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    print(f"Company: {report['company_name']}")
    print(f"Symbol: {report['symbol']}")
    print(f"Analysis Date: {report['analysis_date']}")
    print(f"Current Price: ${report['current_price']:.2f}")
    print(f"Data Points: {report['data_days']} days")
    
    if report['data_quality_warning']:
        print(f"\nWarning: {report['data_quality_warning']}")
    print()
    
    # Performance Analysis
    print(f"\n{'─'*80}")
    print("📊 PERFORMANCE ANALYSIS")
    print(f"{'─'*80}")
    perf = report['performance_analysis']
    print(f"Total Return: {perf['total_return_pct']:.2f}%")
    print(f"Volatility: {perf['volatility_pct']:.2f}%")
    print(f"Current Trend: {perf['current_trend']}")
    print(f"52-Week High: ${perf['52w_high']:.2f}")
    print(f"52-Week Low: ${perf['52w_low']:.2f}")
    print(f"Position in 52W Range: {perf['price_position_in_52w_range']:.1f}%")
    
    # Key Metrics
    print(f"\n{'─'*80}")
    print("📋 KEY METRICS")
    print(f"{'─'*80}")
    metrics = report['key_metrics']
    print(f"P/E Ratio: {metrics['pe_ratio']:.2f}" if metrics['pe_ratio'] else "P/E Ratio: N/A")
    print(f"Market Cap: ${metrics['market_cap']:,.0f}" if metrics['market_cap'] else "Market Cap: N/A")
    print(f"Beta: {metrics['beta']:.2f}" if metrics['beta'] else "Beta: N/A")
    print(f"Dividend Yield: {metrics['dividend_yield']:.2f}%" if metrics['dividend_yield'] else "Dividend Yield: N/A")
    
    # Price Prediction
    print(f"\n{'─'*80}")
    print("🔮 PRICE PREDICTION (30 Days)")
    print(f"{'─'*80}")
    pred = report['price_prediction']
    print(f"Current Price: ${pred['current_price']:.2f}")
    print(f"Predicted Price: ${pred['predicted_price']:.2f}")
    print(f"Expected Change: {pred['predicted_change_pct']:.2f}%")
    print(f"Trend: {pred['trend']}")
    print(f"Confidence Level: {pred['confidence']:.1f}%")
    
    # Market Timing Analysis
    print(f"\n{'─'*80}")
    print("⏰ MARKET TIMING ANALYSIS")
    print(f"{'─'*80}")
    timing = report['market_timing_analysis']
    print(f"Timing Score: {timing['timing_score']:.1f}/100")
    print(f"Interpretation: {timing['interpretation']}\n")
    print("Technical Factors:")
    for i, factor in enumerate(timing['factors'], 1):
        print(f"  {i}. {factor}")
    
    # Corrective Measures
    print(f"\n{'─'*80}")
    print("💡 CORRECTIVE MEASURES & SUGGESTIONS")
    print(f"{'─'*80}")
    for i, suggestion in enumerate(report['corrective_measures'], 1):
        print(f"{i}. {suggestion}")
    
    # Enhanced Technical Signals
    if signal_summary:
        print(f"\n{'─'*80}")
        print("🎯 TECHNICAL SIGNAL ANALYSIS")
        print(f"{'─'*80}")
        print(f"Overall Signal: {signal_summary['overall_signal']}")
        print(f"Signal Strength: {signal_summary['signal_strength']:.2f}")
        print(f"Bullish Signals: {signal_summary['bullish_signals']}")
        print(f"Bearish Signals: {signal_summary['bearish_signals']}")
        print(f"Neutral Signals: {signal_summary['neutral_signals']}")
        print("\nSignal Details:")
        for i, detail in enumerate(signal_summary['signal_details'], 1):
            print(f"  {i}. {detail}")
    
    # Feature Importance (if ML models were used)
    if feature_importance and 'random_forest' in feature_importance:
        print(f"\n{'─'*80}")
        print("🧠 ML MODEL INSIGHTS")
        print(f"{'─'*80}")
        print("Top 5 Most Important Features (Random Forest):")
        for i, (feature, importance) in enumerate(feature_importance['random_forest'][:5], 1):
            print(f"  {i}. {feature}: {importance:.3f}")
    
    # Enhanced Prediction Details
    if 'model_count' in report['price_prediction']:
        print(f"\n{'─'*80}")
        print("🤖 ML ENSEMBLE DETAILS")
        print(f"{'─'*80}")
        pred = report['price_prediction']
        print(f"Models Used: {pred['model_count']}")
        print(f"Best Performing Model: {pred['best_model']}")
        if 'individual_predictions' in pred:
            print("\nIndividual Model Predictions:")
            for model, price in pred['individual_predictions'].items():
                change = (price - pred['current_price']) / pred['current_price'] * 100
                print(f"  {model}: ${price:.2f} ({change:+.2f}%)")
    
    # Disclaimer
    print(f"\n{'─'*80}")
    print("⚠️  DISCLAIMER")
    print(f"{'─'*80}")
    print(report['disclaimer'])
    print(f"{'='*80}\n")


def main():
    """Main application"""
    print("\n🚀 Enhanced Stock Price Prediction + AI Finance Advisor")
    print("Made by Neonite - Version 2.0 with Advanced ML")
    print("=" * 80)
    print("✨ New Features:")
    print("  • Multiple ML models (Random Forest, Gradient Boosting, SVR)")
    print("  • Ensemble predictions with confidence scoring")
    print("  • Advanced technical indicators (20+ indicators)")
    print("  • Feature importance analysis")
    print("  • Enhanced signal analysis")
    print("=" * 80)
    
    # Get ticker input
    ticker = input("\nEnter stock ticker (e.g., GOOGL, AAPL, MSFT): ").strip()
    
    if not ticker:
        ticker = "GOOGL"
        print(f"Using default ticker: {ticker}")
    
    period = input("Enter time period (1mo, 3mo, 6mo, 1y, 2y, 5y) [default: 1y]: ").strip().lower()
    if not period:
        period = "1y"
    
    # Validate period
    valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    if period not in valid_periods:
        print(f"⚠️  Invalid period '{period}'. Using default: 1y")
        period = "1y"
    
    try:
        report = analyze_stock(ticker, period)
        
        if report is None:
            print("\nWould you like to try another ticker?")
        
        # Ask if user wants to analyze another stock
        while True:
            print()
            another = input("Analyze another stock? (y/n): ").strip().lower()
            if another in ['y', 'yes']:
                ticker = input("Enter stock ticker: ").strip()
                if ticker:
                    period_input = input(f"Enter time period [default: {period}]: ").strip().lower()
                    if period_input and period_input in valid_periods:
                        period = period_input
                    report = analyze_stock(ticker, period)
            elif another in ['n', 'no']:
                break
            else:
                print("Please enter 'y' or 'n'")
        
        print("\n" + "="*80)
        print("Thank you for using the AI Finance Advisor!")
        print("Made by Neonite")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        print("Thank you for using the AI Finance Advisor!")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please report this issue if it persists.")


if __name__ == "__main__":
    main()
