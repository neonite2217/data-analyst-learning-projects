"""
Enhanced Stock Price Predictor - Uses multiple ML models for price prediction
Made by Neonite
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """Enhanced predictor with multiple ML models and ensemble methods"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        self.trained_models = {}
        self.feature_importance = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with engineered features
        """
        features = df.copy()
        
        # Price-based features
        features['price_change'] = features['Close'].pct_change()
        features['price_change_2d'] = features['Close'].pct_change(periods=2)
        features['price_change_5d'] = features['Close'].pct_change(periods=5)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(df) > window:
                features[f'sma_{window}'] = features['Close'].rolling(window=window).mean()
                features[f'price_to_sma_{window}'] = features['Close'] / features[f'sma_{window}']
        
        # Volatility features
        features['volatility_5d'] = features['price_change'].rolling(window=5).std()
        features['volatility_20d'] = features['price_change'].rolling(window=20).std()
        
        # Volume features (if available)
        if 'Volume' in features.columns:
            features['volume_change'] = features['Volume'].pct_change()
            features['volume_sma_10'] = features['Volume'].rolling(window=10).mean()
            features['volume_ratio'] = features['Volume'] / features['volume_sma_10']
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(features['Close'])
        macd_data = self._calculate_macd(features['Close'])
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(features['Close'])
        features['bb_upper'] = bb_data['upper']
        features['bb_lower'] = bb_data['lower']
        features['bb_position'] = (features['Close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['Volume'].shift(lag) if 'Volume' in features.columns else 0
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return {'macd': macd, 'signal': signal, 'histogram': histogram}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return {
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
            'middle': sma
        }
    
    def prepare_ml_data(self, df: pd.DataFrame, target_days: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ML models with feature engineering
        
        Args:
            df: DataFrame with stock data
            target_days: Days ahead to predict
        
        Returns:
            X, y arrays for training
        """
        # Create features
        features_df = self.create_features(df)
        
        # Select feature columns (exclude target and non-numeric)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Close', 'Date'] and features_df[col].dtype in ['float64', 'int64']]
        
        # Create target (future price)
        features_df['target'] = features_df['Close'].shift(-target_days)
        
        # Remove rows with NaN values
        clean_data = features_df[feature_cols + ['target']].dropna()
        
        if len(clean_data) < 10:
            raise ValueError("Insufficient data after feature engineering")
        
        X = clean_data[feature_cols].values
        y = clean_data['target'].values
        
        return X, y, feature_cols
    
    def train_models(self, df: pd.DataFrame, target_days: int = 1) -> Dict[str, Any]:
        """
        Train multiple ML models and return performance metrics
        
        Args:
            df: DataFrame with stock data
            target_days: Days ahead to predict
            
        Returns:
            Dictionary with model performance
        """
        try:
            X, y, feature_cols = self.prepare_ml_data(df, target_days)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            model_performance = {}
            
            for name, model in self.models.items():
                try:
                    # Train model
                    if name == 'svr':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    model_performance[name] = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2': r2,
                        'model': model
                    }
                    
                    # Store trained model
                    self.trained_models[name] = model
                    
                    # Feature importance (for tree-based models)
                    if hasattr(model, 'feature_importances_'):
                        importance_dict = dict(zip(feature_cols, model.feature_importances_))
                        self.feature_importance[name] = sorted(
                            importance_dict.items(), key=lambda x: x[1], reverse=True
                        )
                
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            return model_performance
            
        except Exception as e:
            print(f"Error in model training: {e}")
            return {}
    
    def ensemble_prediction(self, df: pd.DataFrame, days_ahead: int = 1) -> Dict[str, Any]:
        """
        Make ensemble prediction using multiple trained models
        
        Args:
            df: DataFrame with stock data
            days_ahead: Days ahead to predict
            
        Returns:
            Dictionary with ensemble prediction results
        """
        try:
            # Train models first
            model_performance = self.train_models(df, days_ahead)
            
            if not model_performance:
                return self.simple_trend_prediction(df, days_ahead)
            
            # Prepare latest data for prediction
            X, _, feature_cols = self.prepare_ml_data(df, days_ahead)
            latest_features = X[-1:] # Get the most recent feature set
            
            current_price = df['Close'].iloc[-1]
            predictions = []
            weights = []
            
            # Get predictions from each model
            for name, perf in model_performance.items():
                try:
                    model = perf['model']
                    
                    if name == 'svr':
                        latest_scaled = self.feature_scaler.transform(latest_features)
                        pred = model.predict(latest_scaled)[0]
                    else:
                        pred = model.predict(latest_features)[0]
                    
                    predictions.append(pred)
                    # Weight by R² score (higher R² gets more weight)
                    weight = max(0.1, perf['r2']) if perf['r2'] > 0 else 0.1
                    weights.append(weight)
                    
                except Exception as e:
                    continue
            
            if not predictions:
                return self.simple_trend_prediction(df, days_ahead)
            
            # Calculate weighted ensemble prediction
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_price = np.average(predictions, weights=weights)
            predicted_change = (ensemble_price - current_price) / current_price
            
            # Calculate confidence based on model agreement
            pred_std = np.std(predictions)
            price_volatility = df['Close'].pct_change().std()
            confidence = max(20, min(95, 80 - (pred_std / current_price * 1000)))
            
            # Determine trend
            trend = "Upward" if predicted_change > 0.01 else "Downward" if predicted_change < -0.01 else "Sideways"
            
            return {
                'current_price': current_price,
                'predicted_price': ensemble_price,
                'predicted_change_pct': predicted_change * 100,
                'trend': trend,
                'confidence': confidence,
                'days_ahead': days_ahead,
                'model_count': len(predictions),
                'best_model': max(model_performance.items(), key=lambda x: x[1]['r2'])[0],
                'ensemble_std': pred_std,
                'individual_predictions': dict(zip(model_performance.keys(), predictions))
            }
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return self.simple_trend_prediction(df, days_ahead)
    
    def simple_trend_prediction(self, df: pd.DataFrame, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Simple trend-based prediction using moving averages and momentum
        
        Args:
            df: DataFrame with stock data
            days_ahead: Number of days to predict ahead
        
        Returns:
            Dictionary with prediction results
        """
        recent_data = df.tail(60)
        current_price = df['Close'].iloc[-1]
        
        # Calculate trend indicators (handle short data)
        window_20 = min(20, len(df))
        window_50 = min(50, len(df))
        sma_20 = df['Close'].rolling(window=window_20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(window=window_50).mean().iloc[-1]
        
        # Calculate momentum (handle short data)
        lookback = min(30, len(df) - 1)
        momentum = (current_price - df['Close'].iloc[-lookback-1]) / df['Close'].iloc[-lookback-1]
        
        # Calculate volatility
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Simple prediction based on trend
        if sma_20 > sma_50:
            trend = "Upward"
            predicted_change = momentum * 0.5  # Conservative estimate
        else:
            trend = "Downward"
            predicted_change = momentum * 0.5
        
        predicted_price = current_price * (1 + predicted_change)
        confidence = max(0, min(100, 70 - (volatility * 100)))
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': predicted_change * 100,
            'trend': trend,
            'confidence': confidence,
            'volatility': volatility,
            'days_ahead': days_ahead,
            'method': 'simple_trend'
        }
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, List]:
        """
        Get feature importance for trained models
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Dictionary of feature importance rankings
        """
        if model_name and model_name in self.feature_importance:
            return {model_name: self.feature_importance[model_name]}
        
        return self.feature_importance
