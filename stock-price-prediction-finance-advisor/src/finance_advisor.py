"""
AI Finance Advisor - Provides analysis and insights (NO buy/sell recommendations)
Made by Neonite
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class FinanceAdvisor:
    """
    AI-powered finance advisor that provides analysis and insights.
    NOTE: Does NOT provide buy/sell recommendations, only educational analysis.
    """
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_stock_performance(self, df: pd.DataFrame, stock_info: Dict) -> Dict[str, Any]:
        """Analyze overall stock performance"""
        current_price = df['Close'].iloc[-1]
        start_price = df['Close'].iloc[0]
        
        # Performance metrics
        total_return = ((current_price - start_price) / start_price) * 100
        
        # Volatility analysis
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Trend analysis
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
        
        # Price position
        high_52w = stock_info.get('52_week_high', current_price)
        low_52w = stock_info.get('52_week_low', current_price)
        price_position = ((current_price - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50
        
        return {
            'total_return_pct': total_return,
            'volatility_pct': volatility,
            'current_trend': 'Upward' if sma_20 > sma_50 else 'Downward',
            'price_position_in_52w_range': price_position,
            'current_price': current_price,
            '52w_high': high_52w,
            '52w_low': low_52w
        }
    
    def assess_market_timing(self, df: pd.DataFrame, technical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess market timing factors (NOT a buy/sell signal)
        Provides educational analysis about current market conditions
        """
        latest = technical_data.iloc[-1]
        
        factors = []
        score = 0.0
        max_score = 0.0
        
        # RSI Analysis (weighted 25%)
        rsi = latest['RSI']
        max_score += 25
        if rsi < 30:
            factors.append(f"RSI at {rsi:.1f} indicates oversold conditions (below 30)")
            score += 25
        elif rsi > 70:
            factors.append(f"RSI at {rsi:.1f} indicates overbought conditions (above 70)")
            score += 5
        elif 40 <= rsi <= 60:
            factors.append(f"RSI at {rsi:.1f} is neutral (balanced)")
            score += 15
        elif rsi < 40:
            factors.append(f"RSI at {rsi:.1f} is slightly oversold")
            score += 20
        else:
            factors.append(f"RSI at {rsi:.1f} is slightly overbought")
            score += 10
        
        # Moving Average Analysis (weighted 25%)
        max_score += 25
        price_vs_sma20 = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
        price_vs_sma50 = ((latest['Close'] - latest['SMA_50']) / latest['SMA_50']) * 100
        
        if latest['Close'] > latest['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
            factors.append(f"Strong uptrend: Price {price_vs_sma50:.1f}% above 50-day MA")
            score += 25
        elif latest['Close'] > latest['SMA_50']:
            factors.append(f"Price {price_vs_sma50:.1f}% above 50-day MA (positive momentum)")
            score += 20
        elif latest['Close'] < latest['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
            factors.append(f"Downtrend: Price {abs(price_vs_sma50):.1f}% below 50-day MA")
            score += 5
        else:
            factors.append(f"Price {abs(price_vs_sma50):.1f}% below 50-day MA (negative momentum)")
            score += 10
        
        # MACD Analysis (weighted 25%)
        max_score += 25
        macd_diff = latest['MACD'] - latest['MACD_Signal']
        if latest['MACD'] > latest['MACD_Signal'] and macd_diff > 0.5:
            factors.append(f"Strong MACD bullish signal (diff: {macd_diff:.2f})")
            score += 25
        elif latest['MACD'] > latest['MACD_Signal']:
            factors.append(f"MACD bullish crossover (diff: {macd_diff:.2f})")
            score += 20
        elif latest['MACD'] < latest['MACD_Signal'] and macd_diff < -0.5:
            factors.append(f"Strong MACD bearish signal (diff: {macd_diff:.2f})")
            score += 5
        else:
            factors.append(f"MACD bearish crossover (diff: {macd_diff:.2f})")
            score += 10
        
        # Bollinger Bands Analysis (weighted 25%)
        max_score += 25
        bb_position = ((latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])) * 100
        
        if bb_position < 20:
            factors.append(f"Price at {bb_position:.0f}% of Bollinger Band range (near support)")
            score += 25
        elif bb_position > 80:
            factors.append(f"Price at {bb_position:.0f}% of Bollinger Band range (near resistance)")
            score += 5
        elif 40 <= bb_position <= 60:
            factors.append(f"Price at {bb_position:.0f}% of Bollinger Band range (neutral)")
            score += 15
        else:
            factors.append(f"Price at {bb_position:.0f}% of Bollinger Band range")
            score += 12
        
        timing_score = (score / max_score) * 100
        
        return {
            'timing_score': timing_score,
            'factors': factors,
            'interpretation': self._interpret_timing_score(timing_score)
        }
    
    def _interpret_timing_score(self, score: float) -> str:
        """Interpret the timing score"""
        if score >= 80:
            return "Strong technical indicators - highly favorable conditions"
        elif score >= 65:
            return "Good technical setup - favorable conditions for consideration"
        elif score >= 50:
            return "Moderate signals - some positive, some negative indicators"
        elif score >= 35:
            return "Weak signals - more caution than opportunity"
        else:
            return "Poor technical setup - significant caution warranted"
    
    def suggest_corrective_measures(self, performance: Dict, stock_info: Dict) -> List[str]:
        """Suggest corrective measures based on analysis"""
        suggestions = []
        
        # Volatility-based suggestions
        if performance['volatility_pct'] > 40:
            suggestions.append("High volatility detected - Consider reviewing position sizing and risk tolerance")
            suggestions.append("Diversification across sectors may help reduce portfolio volatility")
        
        # Performance-based suggestions
        if performance['total_return_pct'] < -15:
            suggestions.append("Significant drawdown observed - Review investment thesis and fundamentals")
            suggestions.append("Consider dollar-cost averaging if fundamentals remain strong")
        
        # Valuation suggestions
        pe_ratio = stock_info.get('pe_ratio', 0)
        if pe_ratio > 30:
            suggestions.append(f"P/E ratio of {pe_ratio:.1f} is elevated - Monitor valuation metrics")
        elif pe_ratio > 0 and pe_ratio < 15:
            suggestions.append(f"P/E ratio of {pe_ratio:.1f} appears reasonable relative to market")
        
        # Debt analysis
        debt_to_equity = stock_info.get('debt_to_equity', 0)
        if debt_to_equity > 2:
            suggestions.append("High debt-to-equity ratio - Monitor company's financial health")
        
        # Position in range
        if performance['price_position_in_52w_range'] < 20:
            suggestions.append("Stock near 52-week low - Research if fundamentals justify current price")
        elif performance['price_position_in_52w_range'] > 80:
            suggestions.append("Stock near 52-week high - Consider if valuation is stretched")
        
        if not suggestions:
            suggestions.append("Current metrics appear balanced - Continue monitoring regularly")
        
        return suggestions
    
    def generate_comprehensive_report(self, stock_symbol: str, df: pd.DataFrame, 
                                     stock_info: Dict, technical_data: pd.DataFrame,
                                     prediction: Dict) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        performance = self.analyze_stock_performance(df, stock_info)
        timing = self.assess_market_timing(df, technical_data)
        suggestions = self.suggest_corrective_measures(performance, stock_info)
        
        # Add data quality warning
        data_quality_warning = None
        if len(df) < 30:
            data_quality_warning = f"Limited data: Only {len(df)} days available. Analysis may be less reliable. Recommend using 3+ months of data."
        elif len(df) < 60:
            data_quality_warning = f"Moderate data: {len(df)} days available. Consider using 3+ months for better accuracy."
        
        report = {
            'symbol': stock_symbol,
            'company_name': stock_info.get('name', 'N/A'),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': performance['current_price'],
            'data_days': len(df),
            'data_quality_warning': data_quality_warning,
            'performance_analysis': performance,
            'market_timing_analysis': timing,
            'price_prediction': prediction,
            'corrective_measures': suggestions,
            'key_metrics': {
                'pe_ratio': stock_info.get('pe_ratio', 0),
                'market_cap': stock_info.get('market_cap', 0),
                'beta': stock_info.get('beta', 0),
                'dividend_yield': stock_info.get('dividend_yield', 0)  # Already converted to percentage in fetcher
            },
            'disclaimer': 'This is educational analysis only. Not financial advice. Consult a licensed financial advisor before making investment decisions.'
        }
        
        return report
