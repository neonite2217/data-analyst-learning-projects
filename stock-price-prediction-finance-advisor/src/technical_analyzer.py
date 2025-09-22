"""
Technical Analysis Module - Calculates technical indicators and patterns
Made by Neonite
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class TechnicalAnalyzer:
    """Performs technical analysis on stock data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def calculate_sma(self, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average"""
        return self.df['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return self.df['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def calculate_volatility(self, period: int = 30) -> float:
        """Calculate historical volatility"""
        returns = self.df['Close'].pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)
        return volatility.iloc[-1] if not volatility.empty else 0.0
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = self.df['Low'].rolling(window=k_period).min()
        high_max = self.df['High'].rolling(window=k_period).max()
        
        k_percent = 100 * ((self.df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {'%K': k_percent, '%D': d_percent}
    
    def calculate_williams_r(self, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = self.df['High'].rolling(window=period).max()
        low_min = self.df['Low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - self.df['Close']) / (high_max - low_min))
        return williams_r
    
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index"""
        high_diff = self.df['High'].diff()
        low_diff = self.df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr = self.calculate_atr(period)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {'ADX': adx, '+DI': plus_di, '-DI': minus_di}
    
    def calculate_cci(self, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume"""
        if 'Volume' not in self.df.columns:
            return pd.Series([0] * len(self.df), index=self.df.index)
        
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['Close'].iloc[i] > self.df['Close'].iloc[i-1]:
                obv.append(obv[-1] + self.df['Volume'].iloc[i])
            elif self.df['Close'].iloc[i] < self.df['Close'].iloc[i-1]:
                obv.append(obv[-1] - self.df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=self.df.index)
    
    def calculate_fibonacci_levels(self, period: int = 50) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        recent_data = self.df.tail(period)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        diff = high - low
        
        levels = {
            'high': high,
            'low': low,
            'fib_23.6': high - 0.236 * diff,
            'fib_38.2': high - 0.382 * diff,
            'fib_50.0': high - 0.500 * diff,
            'fib_61.8': high - 0.618 * diff,
            'fib_78.6': high - 0.786 * diff
        }
        
        return levels
    
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect common chart patterns"""
        patterns = {}
        
        # Simple pattern detection
        recent_closes = self.df['Close'].tail(20)
        
        # Double top/bottom detection (simplified)
        if len(recent_closes) >= 10:
            peaks = []
            troughs = []
            
            for i in range(1, len(recent_closes) - 1):
                if (recent_closes.iloc[i] > recent_closes.iloc[i-1] and 
                    recent_closes.iloc[i] > recent_closes.iloc[i+1]):
                    peaks.append((i, recent_closes.iloc[i]))
                elif (recent_closes.iloc[i] < recent_closes.iloc[i-1] and 
                      recent_closes.iloc[i] < recent_closes.iloc[i+1]):
                    troughs.append((i, recent_closes.iloc[i]))
            
            patterns['peaks_count'] = len(peaks)
            patterns['troughs_count'] = len(troughs)
            
            # Head and shoulders detection (very basic)
            if len(peaks) >= 3:
                patterns['potential_head_shoulders'] = True
            else:
                patterns['potential_head_shoulders'] = False
        
        # Support and resistance levels
        patterns['support_level'] = self.df['Low'].tail(50).min()
        patterns['resistance_level'] = self.df['High'].tail(50).max()
        
        return patterns
    
    def get_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators and return as DataFrame"""
        result = self.df.copy()
        
        # Basic indicators
        result['SMA_20'] = self.calculate_sma(20)
        result['SMA_50'] = self.calculate_sma(50)
        result['EMA_20'] = self.calculate_ema(20)
        result['RSI'] = self.calculate_rsi()
        
        # MACD
        macd_data = self.calculate_macd()
        result['MACD'] = macd_data['macd']
        result['MACD_Signal'] = macd_data['signal']
        result['MACD_Histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands()
        result['BB_Upper'] = bb_data['upper']
        result['BB_Middle'] = bb_data['middle']
        result['BB_Lower'] = bb_data['lower']
        
        # Advanced indicators
        stoch_data = self.calculate_stochastic()
        result['Stoch_K'] = stoch_data['%K']
        result['Stoch_D'] = stoch_data['%D']
        
        result['Williams_R'] = self.calculate_williams_r()
        result['ATR'] = self.calculate_atr()
        result['CCI'] = self.calculate_cci()
        result['OBV'] = self.calculate_obv()
        
        # ADX indicators
        adx_data = self.calculate_adx()
        result['ADX'] = adx_data['ADX']
        result['Plus_DI'] = adx_data['+DI']
        result['Minus_DI'] = adx_data['-DI']
        
        return result
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get a summary of all technical signals"""
        indicators = self.get_all_indicators()
        latest = indicators.iloc[-1]
        
        signals = {
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'signal_details': []
        }
        
        # RSI signals
        if latest['RSI'] < 30:
            signals['bullish_signals'] += 1
            signals['signal_details'].append("RSI oversold (bullish)")
        elif latest['RSI'] > 70:
            signals['bearish_signals'] += 1
            signals['signal_details'].append("RSI overbought (bearish)")
        else:
            signals['neutral_signals'] += 1
            signals['signal_details'].append("RSI neutral")
        
        # MACD signals
        if latest['MACD'] > latest['MACD_Signal']:
            signals['bullish_signals'] += 1
            signals['signal_details'].append("MACD bullish crossover")
        else:
            signals['bearish_signals'] += 1
            signals['signal_details'].append("MACD bearish crossover")
        
        # Moving average signals
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals['bullish_signals'] += 1
            signals['signal_details'].append("Price above moving averages (bullish)")
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            signals['bearish_signals'] += 1
            signals['signal_details'].append("Price below moving averages (bearish)")
        else:
            signals['neutral_signals'] += 1
            signals['signal_details'].append("Mixed moving average signals")
        
        # Stochastic signals
        if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
            signals['bullish_signals'] += 1
            signals['signal_details'].append("Stochastic oversold (bullish)")
        elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
            signals['bearish_signals'] += 1
            signals['signal_details'].append("Stochastic overbought (bearish)")
        else:
            signals['neutral_signals'] += 1
            signals['signal_details'].append("Stochastic neutral")
        
        # Overall signal
        total_signals = signals['bullish_signals'] + signals['bearish_signals'] + signals['neutral_signals']
        if signals['bullish_signals'] > signals['bearish_signals']:
            signals['overall_signal'] = 'BULLISH'
        elif signals['bearish_signals'] > signals['bullish_signals']:
            signals['overall_signal'] = 'BEARISH'
        else:
            signals['overall_signal'] = 'NEUTRAL'
        
        signals['signal_strength'] = max(signals['bullish_signals'], signals['bearish_signals']) / total_signals
        
        return signals
