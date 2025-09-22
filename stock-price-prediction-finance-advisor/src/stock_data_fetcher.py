"""
Stock Data Fetcher - Retrieves stock data from Yahoo Finance (Google Finance data)
Made by Neonite
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple


class StockDataFetcher:
    """Fetches and processes stock data from Yahoo Finance"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()  # Case-insensitive and trim whitespace
        self.stock = yf.Ticker(self.ticker)
        self._validate_ticker()
        
    def get_historical_data(self, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with historical stock data
        """
        df = self.stock.history(period=period)
        return df
    
    def get_stock_info(self) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        try:
            info = self.stock.info
            return {
                'symbol': info.get('symbol', self.ticker),
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,  # Convert to percentage
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
            }
        except Exception as e:
            print(f"Error fetching stock info: {e}")
            return {}
    
    def get_current_price(self) -> float:
        """Get the most recent stock price"""
        try:
            data = self.stock.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
            return 0.0
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return 0.0
    
    def _validate_ticker(self) -> bool:
        """Validate if ticker exists and has data"""
        try:
            info = self.stock.info
            # Check if we got valid data
            if not info or 'symbol' not in info:
                raise ValueError(f"Invalid ticker symbol: {self.ticker}")
            return True
        except Exception:
            # Ticker might still be valid, just no info available
            return True
    
    @staticmethod
    def search_ticker(query: str) -> List[Tuple[str, str]]:
        """
        Search for stock tickers by company name or symbol
        
        Args:
            query: Search term (company name or ticker)
        
        Returns:
            List of tuples (ticker, company_name)
        """
        try:
            query = query.upper().strip()
            ticker = yf.Ticker(query)
            info = ticker.info
            
            if info and 'symbol' in info:
                return [(info.get('symbol', query), info.get('longName', 'N/A'))]
            return []
        except Exception:
            return []
    
    @staticmethod
    def validate_ticker_exists(ticker: str) -> Tuple[bool, str]:
        """
        Validate if a ticker exists and return error message if not
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ticker = ticker.upper().strip()
            stock = yf.Ticker(ticker)
            
            # Try to fetch recent data
            hist = stock.history(period="5d")
            
            if hist.empty:
                return False, f"No data available for ticker '{ticker}'. Please verify the symbol."
            
            # Try to get basic info
            info = stock.info
            if not info or len(info) < 5:
                return False, f"Ticker '{ticker}' appears to be invalid or delisted."
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating ticker '{ticker}': {str(e)}"
