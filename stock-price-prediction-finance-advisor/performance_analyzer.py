#!/usr/bin/env python3
"""
Performance Analysis Tool - Backtesting and model evaluation
Made by Neonite
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stock_data_fetcher import StockDataFetcher
from technical_analyzer import TechnicalAnalyzer
from stock_predictor import StockPredictor

class PerformanceAnalyzer:
    """Analyze prediction accuracy and model performance"""
    
    def __init__(self):
        self.results = []
    
    def backtest_predictions(self, ticker: str, days_back: int = 60, prediction_days: int = 7):
        """
        Backtest predictions by making historical predictions and comparing to actual prices
        
        Args:
            ticker: Stock symbol
            days_back: How many days back to test
            prediction_days: Days ahead to predict
        """
        print(f"\n🔍 Backtesting {ticker} - {days_back} days back, {prediction_days} day predictions")
        print("-" * 70)
        
        try:
            # Get extended historical data
            fetcher = StockDataFetcher(ticker)
            full_df = fetcher.get_historical_data(period="1y")
            
            if len(full_df) < days_back + prediction_days + 30:
                print(f"❌ Insufficient data for backtesting")
                return None
            
            predictions = []
            actuals = []
            dates = []
            
            # Walk forward through time making predictions
            for i in range(days_back, 0, -5):  # Test every 5 days
                # Get data up to this point
                end_idx = len(full_df) - i
                historical_data = full_df.iloc[:end_idx]
                
                # Make prediction
                predictor = StockPredictor()
                try:
                    pred_result = predictor.ensemble_prediction(historical_data, days_ahead=prediction_days)
                    
                    # Get actual price after prediction_days
                    actual_idx = end_idx + prediction_days - 1
                    if actual_idx < len(full_df):
                        actual_price = full_df.iloc[actual_idx]['Close']
                        
                        predictions.append(pred_result['predicted_price'])
                        actuals.append(actual_price)
                        dates.append(full_df.index[end_idx])
                        
                        # Calculate accuracy
                        pred_change = (pred_result['predicted_price'] - pred_result['current_price']) / pred_result['current_price']
                        actual_change = (actual_price - pred_result['current_price']) / pred_result['current_price']
                        
                        direction_correct = (pred_change > 0) == (actual_change > 0)
                        
                        print(f"Date: {full_df.index[end_idx].strftime('%Y-%m-%d')} | "
                              f"Pred: ${pred_result['predicted_price']:.2f} | "
                              f"Actual: ${actual_price:.2f} | "
                              f"Direction: {'✓' if direction_correct else '✗'}")
                        
                except Exception as e:
                    continue
            
            if predictions:
                return self.calculate_backtest_metrics(predictions, actuals, dates)
            else:
                print("❌ No successful predictions made")
                return None
                
        except Exception as e:
            print(f"❌ Error in backtesting: {e}")
            return None
    
    def calculate_backtest_metrics(self, predictions, actuals, dates):
        """Calculate comprehensive backtest metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Basic metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Direction accuracy
        pred_changes = np.diff(predictions)
        actual_changes = np.diff(actuals[:-1])  # Align arrays
        direction_accuracy = np.mean((pred_changes > 0) == (actual_changes > 0)) * 100
        
        # Correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'correlation': correlation,
            'total_predictions': len(predictions),
            'avg_prediction': np.mean(predictions),
            'avg_actual': np.mean(actuals)
        }
        
        print(f"\n📊 Backtest Results:")
        print(f"  Total Predictions: {metrics['total_predictions']}")
        print(f"  Mean Absolute Error: ${metrics['mae']:.2f}")
        print(f"  Root Mean Square Error: ${metrics['rmse']:.2f}")
        print(f"  Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
        print(f"  Correlation: {metrics['correlation']:.3f}")
        
        return metrics
    
    def analyze_model_stability(self, ticker: str, iterations: int = 10):
        """Test model stability across multiple runs"""
        print(f"\n🔄 Model Stability Analysis - {ticker} ({iterations} iterations)")
        print("-" * 60)
        
        try:
            fetcher = StockDataFetcher(ticker)
            df = fetcher.get_historical_data(period="6mo")
            
            predictions = []
            confidences = []
            
            for i in range(iterations):
                predictor = StockPredictor()
                result = predictor.ensemble_prediction(df, days_ahead=7)
                
                predictions.append(result['predicted_price'])
                confidences.append(result['confidence'])
                
                print(f"  Run {i+1}: ${result['predicted_price']:.2f} (confidence: {result['confidence']:.1f}%)")
            
            # Calculate stability metrics
            pred_std = np.std(predictions)
            pred_mean = np.mean(predictions)
            cv = (pred_std / pred_mean) * 100  # Coefficient of variation
            
            conf_mean = np.mean(confidences)
            conf_std = np.std(confidences)
            
            print(f"\n📈 Stability Metrics:")
            print(f"  Prediction Mean: ${pred_mean:.2f}")
            print(f"  Prediction Std Dev: ${pred_std:.2f}")
            print(f"  Coefficient of Variation: {cv:.2f}%")
            print(f"  Average Confidence: {conf_mean:.1f}% ± {conf_std:.1f}%")
            
            stability_score = max(0, 100 - cv)  # Lower CV = higher stability
            print(f"  Stability Score: {stability_score:.1f}/100")
            
            return {
                'prediction_mean': pred_mean,
                'prediction_std': pred_std,
                'coefficient_variation': cv,
                'confidence_mean': conf_mean,
                'confidence_std': conf_std,
                'stability_score': stability_score
            }
            
        except Exception as e:
            print(f"❌ Error in stability analysis: {e}")
            return None
    
    def compare_stocks_performance(self, tickers: list):
        """Compare prediction performance across multiple stocks"""
        print(f"\n🏆 Multi-Stock Performance Comparison")
        print("=" * 70)
        
        results = {}
        
        for ticker in tickers:
            print(f"\n📊 Analyzing {ticker}...")
            
            # Backtest
            backtest_result = self.backtest_predictions(ticker, days_back=30, prediction_days=5)
            
            # Stability
            stability_result = self.analyze_model_stability(ticker, iterations=5)
            
            if backtest_result and stability_result:
                results[ticker] = {
                    'backtest': backtest_result,
                    'stability': stability_result
                }
        
        # Summary comparison
        if results:
            print(f"\n🎯 Performance Summary:")
            print("-" * 70)
            print(f"{'Stock':<8} {'Direction%':<12} {'MAPE%':<10} {'Stability':<12} {'Correlation':<12}")
            print("-" * 70)
            
            for ticker, data in results.items():
                direction_acc = data['backtest']['direction_accuracy']
                mape = data['backtest']['mape']
                stability = data['stability']['stability_score']
                correlation = data['backtest']['correlation']
                
                print(f"{ticker:<8} {direction_acc:<12.1f} {mape:<10.1f} {stability:<12.1f} {correlation:<12.3f}")
        
        return results

def main():
    """Main performance analysis"""
    analyzer = PerformanceAnalyzer()
    
    print("🚀 Stock Predictor Performance Analysis")
    print("=" * 70)
    
    # Test individual stock
    test_ticker = "AAPL"
    
    # Backtest
    analyzer.backtest_predictions(test_ticker, days_back=40, prediction_days=7)
    
    # Stability analysis
    analyzer.analyze_model_stability(test_ticker, iterations=8)
    
    # Multi-stock comparison
    test_stocks = ["AAPL", "GOOGL", "MSFT"]
    analyzer.compare_stocks_performance(test_stocks)
    
    print(f"\n{'='*70}")
    print("✅ Performance analysis complete!")
    print("\nKey Insights:")
    print("• Direction accuracy shows how often we predict the right trend")
    print("• MAPE shows average percentage error in price predictions")
    print("• Stability score indicates model consistency across runs")
    print("• Correlation measures how well predictions track actual prices")

if __name__ == "__main__":
    main()