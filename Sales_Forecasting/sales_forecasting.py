import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

warnings.filterwarnings('ignore')

class SalesForecastingApp:
    """
    Sales Forecasting Application with data import, analysis, and export capabilities
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}
        self.config = {
            'target_column': 'monthly_sales',
            'date_column': 'date',
            'categorical_columns': ['store_id', 'product_category'],
            'test_size': 0.2,
            'random_state': 42
        }
        
    def generate_demo_data(self):
        """Generate synthetic sales data for demonstration"""
        print("Generating demo sales data...")
        
        # Date range
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D', 'Store_E']
        categories = ['Smartphones', 'Laptops', 'Tablets', 'Audio', 'Gaming']
        
        data = []
        np.random.seed(42)
        
        current_date = start_date
        while current_date <= end_date:
            for store in stores:
                for category in categories:
                    # Base sales with seasonality
                    base_sales = np.random.normal(50000, 15000)
                    
                    # Seasonal patterns
                    month = current_date.month
                    if month in [11, 12]:  # Holiday season
                        seasonal_multiplier = 1.4
                        holiday_season = 1
                    elif month in [6, 7, 8]:  # Summer
                        seasonal_multiplier = 0.8
                        holiday_season = 0
                    else:
                        seasonal_multiplier = 1.0
                        holiday_season = 0
                    
                    # Store-specific patterns
                    store_multipliers = {'Store_A': 1.2, 'Store_B': 1.0, 'Store_C': 0.9, 
                                       'Store_D': 1.1, 'Store_E': 0.95}
                    
                    # Category-specific patterns
                    category_multipliers = {'Smartphones': 1.3, 'Laptops': 1.1, 'Tablets': 0.8,
                                          'Audio': 0.9, 'Gaming': 1.2}
                    
                    # Promotion (20% chance)
                    promotion_active = np.random.choice([0, 1], p=[0.8, 0.2])
                    promotion_multiplier = 1.15 if promotion_active else 1.0
                    
                    # Calculate final sales
                    monthly_sales = max(base_sales * seasonal_multiplier * 
                                      store_multipliers[store] * 
                                      category_multipliers[category] * 
                                      promotion_multiplier, 10000)
                    
                    units_sold = int(monthly_sales / np.random.uniform(300, 800))
                    
                    data.append({
                        'date': current_date,
                        'store_id': store,
                        'product_category': category,
                        'monthly_sales': round(monthly_sales, 2),
                        'units_sold': units_sold,
                        'promotion_active': promotion_active,
                        'holiday_season': holiday_season,
                        'avg_temperature': np.random.normal(20, 10),
                        'local_unemployment': np.random.normal(5.5, 1.2),
                        'competitor_stores': np.random.randint(2, 8)
                    })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        self.data = pd.DataFrame(data)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        print(f"✓ Demo data generated: {len(self.data)} records")
        return self.data
    
    def load_custom_data(self, file_path):
        """Load custom dataset from file"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
            # Convert date column
            if self.config['date_column'] in self.data.columns:
                self.data[self.config['date_column']] = pd.to_datetime(self.data[self.config['date_column']])
            
            print(f"✓ Custom data loaded: {len(self.data)} records")
            return self.data
        
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.data is None:
            print("✗ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\n📊 Dataset Overview:")
        print(f"Shape: {self.data.shape}")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Stores: {self.data['store_id'].nunique()}")
        print(f"Categories: {self.data['product_category'].nunique()}")
        
        # Missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️ Missing Values:")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values found")
        
        # Summary statistics
        print("\n📈 Sales Statistics:")
        sales_stats = self.data['monthly_sales'].describe()
        print(sales_stats)
        
        # Top performers
        print("\n🏆 Top Performing Stores:")
        store_performance = self.data.groupby('store_id')['monthly_sales'].mean().sort_values(ascending=False)
        print(store_performance.head())
        
        print("\n🏆 Top Performing Categories:")
        category_performance = self.data.groupby('product_category')['monthly_sales'].mean().sort_values(ascending=False)
        print(category_performance.head())
        
        return self.data.describe()
    
    def feature_engineering(self):
        """Create features for modeling"""
        if self.data is None:
            print("✗ No data loaded.")
            return
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        df = self.data.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Lag features (previous month sales by store-category)
        df = df.sort_values(['store_id', 'product_category', 'date'])
        df['sales_lag_1'] = df.groupby(['store_id', 'product_category'])['monthly_sales'].shift(1)
        df['sales_lag_3'] = df.groupby(['store_id', 'product_category'])['monthly_sales'].shift(3)
        
        # Rolling averages
        df['sales_rolling_3'] = df.groupby(['store_id', 'product_category'])['monthly_sales'].rolling(3).mean().values
        df['sales_rolling_6'] = df.groupby(['store_id', 'product_category'])['monthly_sales'].rolling(6).mean().values
        
        # Seasonal indicators
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Category and store encoding
        le_store = LabelEncoder()
        le_category = LabelEncoder()
        
        df['store_encoded'] = le_store.fit_transform(df['store_id'])
        df['category_encoded'] = le_category.fit_transform(df['product_category'])
        
        # Interaction features
        df['store_category_interaction'] = df['store_encoded'] * df['category_encoded']
        
        # Remove rows with NaN values from lag features
        df = df.dropna()
        
        self.processed_data = df
        
        print(f"✓ Features created. Dataset shape: {df.shape}")
        print(f"✓ Features: {list(df.columns)}")
        
        return df
    
    def build_models(self):
        """Build and train predictive models"""
        if self.processed_data is None:
            print("✗ No processed data. Run feature_engineering() first.")
            return
        
        print("\n" + "="*50)
        print("MODEL BUILDING")
        print("="*50)
        
        # Prepare features
        feature_columns = [
            'year', 'month', 'quarter', 'units_sold', 'promotion_active',
            'avg_temperature', 'local_unemployment', 'competitor_stores',
            'sales_lag_1', 'sales_lag_3', 'sales_rolling_3', 'sales_rolling_6',
            'is_holiday_season', 'is_summer', 'store_encoded', 'category_encoded',
            'store_category_interaction'
        ]
        
        X = self.processed_data[feature_columns]
        y = self.processed_data[self.config['target_column']]
        
        # Split data chronologically
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model 1: Linear Regression
        print("\n🔧 Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        # Model 2: Random Forest
        print("🔧 Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=self.config['random_state'])
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Store models and predictions
        self.models = {
            'linear_regression': lr_model,
            'random_forest': rf_model,
            'scaler': scaler
        }
        
        self.predictions = {
            'linear_regression': lr_pred,
            'random_forest': rf_pred,
            'actual': y_test.values,
            'test_dates': self.processed_data.iloc[split_idx:]['date'].values
        }
        
        # Feature importance for Random Forest
        self.feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
        
        # Model evaluation
        self.evaluate_models()
        
        print("✓ Models trained successfully")
    
    def evaluate_models(self):
        """Evaluate model performance"""
        if not self.predictions:
            print("✗ No predictions available. Train models first.")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        actual = self.predictions['actual']
        
        models_to_eval = ['linear_regression', 'random_forest']
        results = []
        
        for model_name in models_to_eval:
            pred = self.predictions[model_name]
            
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            results.append({
                'Model': model_name.replace('_', ' ').title(),
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'R²': round(r2, 4),
                'MAPE (%)': round(mape, 2)
            })
            
            print(f"\n📊 {model_name.replace('_', ' ').title()}:")
            print(f"   MAE: ${mae:,.2f}")
            print(f"   RMSE: ${rmse:,.2f}")
            print(f"   R²: {r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")
        
        # Best model
        best_model = min(results, key=lambda x: x['RMSE'])
        print(f"\n🏆 Best Model: {best_model['Model']} (Lowest RMSE)")
        
        return pd.DataFrame(results)
    
    def generate_visualizations(self):
        """Create comprehensive visualizations"""
        if self.data is None:
            print("✗ No data available for visualization.")
            return
        
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 12)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle('Sales Forecasting Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Sales trend over time
        monthly_sales = self.data.groupby('date')['monthly_sales'].sum()
        axes[0, 0].plot(monthly_sales.index, monthly_sales.values, linewidth=2, color='#2E8B57')
        axes[0, 0].set_title('Monthly Sales Trend', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sales by store
        store_sales = self.data.groupby('store_id')['monthly_sales'].mean().sort_values(ascending=True)
        axes[0, 1].barh(store_sales.index, store_sales.values, color='#FF6B6B')
        axes[0, 1].set_title('Average Sales by Store', fontweight='bold')
        axes[0, 1].set_xlabel('Average Sales ($)')
        
        # 3. Sales by category
        category_sales = self.data.groupby('product_category')['monthly_sales'].mean().sort_values(ascending=False)
        axes[0, 2].bar(category_sales.index, category_sales.values, color='#4ECDC4')
        axes[0, 2].set_title('Average Sales by Category', fontweight='bold')
        axes[0, 2].set_ylabel('Average Sales ($)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Seasonal patterns
        seasonal_data = self.data.groupby(self.data['date'].dt.month)['monthly_sales'].mean()
        axes[1, 0].plot(seasonal_data.index, seasonal_data.values, marker='o', linewidth=2, color='#FF9F40')
        axes[1, 0].set_title('Seasonal Sales Pattern', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Sales ($)')
        axes[1, 0].set_xticks(range(1, 13))
        
        # 5. Promotion impact
        promo_impact = self.data.groupby('promotion_active')['monthly_sales'].mean()
        axes[1, 1].bar(['No Promotion', 'With Promotion'], promo_impact.values, 
                       color=['#FF6B6B', '#4ECDC4'])
        axes[1, 1].set_title('Promotion Impact on Sales', fontweight='bold')
        axes[1, 1].set_ylabel('Average Sales ($)')
        
        # 6. Feature importance (if models are trained)
        if self.feature_importance:
            top_features = dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:8])
            axes[1, 2].barh(list(top_features.keys()), list(top_features.values()), color='#9B59B6')
            axes[1, 2].set_title('Feature Importance (Random Forest)', fontweight='bold')
            axes[1, 2].set_xlabel('Importance')
        else:
            axes[1, 2].text(0.5, 0.5, 'Train models first\nto see feature importance', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Feature Importance', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('sales_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Dashboard saved as 'sales_analysis_dashboard.png'")
        
        # Interactive Plotly visualizations
        self.create_interactive_charts()
        
        plt.show()
    
    def create_interactive_charts(self):
        """Create interactive Plotly charts"""
        print("📊 Creating interactive charts...")
        
        # 1. Time series chart with store breakdown
        fig1 = px.line(self.data, x='date', y='monthly_sales', color='store_id',
                      title='Sales Trend by Store (Interactive)',
                      labels={'monthly_sales': 'Sales ($)', 'date': 'Date'})
        fig1.write_html('sales_trend_interactive.html')
        
        # 2. Sunburst chart for sales hierarchy
        fig2 = px.sunburst(self.data, path=['store_id', 'product_category'], 
                          values='monthly_sales',
                          title='Sales Distribution by Store and Category')
        fig2.write_html('sales_sunburst.html')
        
        # 3. Prediction vs Actual (if models are trained)
        if self.predictions:
            pred_df = pd.DataFrame({
                'Date': self.predictions['test_dates'],
                'Actual': self.predictions['actual'],
                'Linear Regression': self.predictions['linear_regression'],
                'Random Forest': self.predictions['random_forest']
            })
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Actual'], 
                                    name='Actual', mode='lines', line=dict(color='blue')))
            fig3.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Linear Regression'], 
                                    name='Linear Regression', mode='lines', line=dict(color='red')))
            fig3.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Random Forest'], 
                                    name='Random Forest', mode='lines', line=dict(color='green')))
            
            fig3.update_layout(title='Model Predictions vs Actual Sales',
                             xaxis_title='Date', yaxis_title='Sales ($)')
            fig3.write_html('predictions_comparison.html')
        
        print("✓ Interactive charts saved as HTML files")
    
    def export_results(self):
        """Export results to Excel and Power BI compatible formats"""
        if self.data is None:
            print("✗ No data to export.")
            return
        
        print("\n" + "="*50)
        print("EXPORTING RESULTS")
        print("="*50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Excel export with multiple sheets
        excel_filename = f'sales_forecasting_results_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Raw data
            self.data.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # Processed data (if available)
            if self.processed_data is not None:
                self.processed_data.to_excel(writer, sheet_name='Processed_Data', index=False)
            
            # Summary statistics
            summary_stats = self.data.groupby(['store_id', 'product_category']).agg({
                'monthly_sales': ['mean', 'sum', 'std'],
                'units_sold': ['mean', 'sum'],
                'promotion_active': 'mean'
            }).round(2)
            summary_stats.to_excel(writer, sheet_name='Summary_Stats')
            
            # Model evaluation (if available)
            if self.predictions:
                eval_results = self.evaluate_models()
                eval_results.to_excel(writer, sheet_name='Model_Performance', index=False)
                
                # Predictions
                pred_df = pd.DataFrame({
                    'Date': self.predictions['test_dates'],
                    'Actual_Sales': self.predictions['actual'],
                    'LinearRegression_Pred': self.predictions['linear_regression'],
                    'RandomForest_Pred': self.predictions['random_forest']
                })
                pred_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Feature importance (if available)
            if self.feature_importance:
                importance_df = pd.DataFrame(list(self.feature_importance.items()), 
                                           columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=False)
                importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        print(f"✓ Excel file exported: {excel_filename}")
        
        # 2. Power BI compatible CSV files
        powerbi_folder = f'powerbi_exports_{timestamp}'
        os.makedirs(powerbi_folder, exist_ok=True)
        
        # Main dataset for Power BI
        powerbi_data = self.data.copy()
        powerbi_data['year_month'] = powerbi_data['date'].dt.to_period('M').astype(str)
        powerbi_data.to_csv(f'{powerbi_folder}/sales_data_powerbi.csv', index=False)
        
        # Aggregated monthly data
        monthly_agg = self.data.groupby(['date', 'store_id', 'product_category']).agg({
            'monthly_sales': 'sum',
            'units_sold': 'sum',
            'promotion_active': 'max',
            'holiday_season': 'max'
        }).reset_index()
        monthly_agg.to_csv(f'{powerbi_folder}/monthly_aggregated.csv', index=False)
        
        # Store performance summary
        store_summary = self.data.groupby('store_id').agg({
            'monthly_sales': ['mean', 'sum', 'std'],
            'units_sold': 'sum',
            'promotion_active': 'mean'
        }).round(2)
        store_summary.columns = ['_'.join(col).strip() for col in store_summary.columns]
        store_summary.reset_index().to_csv(f'{powerbi_folder}/store_performance.csv', index=False)
        
        # Category performance summary
        category_summary = self.data.groupby('product_category').agg({
            'monthly_sales': ['mean', 'sum', 'std'],
            'units_sold': 'sum',
            'promotion_active': 'mean'
        }).round(2)
        category_summary.columns = ['_'.join(col).strip() for col in category_summary.columns]
        category_summary.reset_index().to_csv(f'{powerbi_folder}/category_performance.csv', index=False)
        
        print(f"✓ Power BI files exported to folder: {powerbi_folder}")
        
        # 3. JSON metadata for API integration
        metadata = {
            'export_timestamp': timestamp,
            'data_shape': list(self.data.shape),
            'date_range': {
                'start': str(self.data['date'].min()),
                'end': str(self.data['date'].max())
            },
            'stores': list(self.data['store_id'].unique()),
            'categories': list(self.data['product_category'].unique()),
            'model_performance': {}
        }
        
        if self.predictions:
            metadata['model_performance'] = {
                'models_trained': list(self.models.keys()),
                'test_samples': len(self.predictions['actual']),
                'best_model_rmse': float(np.sqrt(mean_squared_error(
                    self.predictions['actual'], 
                    self.predictions['random_forest']
                )))
            }
        
        with open(f'export_metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print("✓ Metadata JSON created")
        print("\n📁 Export Summary:")
        print(f"   • Excel workbook: {excel_filename}")
        print(f"   • Power BI folder: {powerbi_folder}")
        print(f"   • Interactive HTML charts: sales_trend_interactive.html, sales_sunburst.html")
        print(f"   • Dashboard image: sales_analysis_dashboard.png")
        
        return excel_filename, powerbi_folder
    
    def run_complete_analysis(self, use_demo_data=True, custom_data_path=None):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Sales Forecasting Analysis Pipeline")
        print("="*60)
        
        try:
            # Step 1: Load data
            if use_demo_data:
                self.generate_demo_data()
            else:
                if custom_data_path:
                    self.load_custom_data(custom_data_path)
                else:
                    print("✗ No data path provided for custom data")
                    return
            
            # Step 2: Explore data
            self.explore_data()
            
            # Step 3: Feature engineering
            self.feature_engineering()
            
            # Step 4: Build models
            self.build_models()
            
            # Step 5: Generate visualizations
            self.generate_visualizations()
            
            # Step 6: Export results
            self.export_results()
            
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETE!")
            print("="*60)
            print("📊 Check the generated files:")
            print("   • Excel workbook with all results")
            print("   • Power BI compatible CSV files")
            print("   • Interactive HTML charts")
            print("   • Static dashboard image")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            return False
        
        return True

# Example usage and demo
if __name__ == "__main__":
    # Initialize the application
    app = SalesForecastingApp()
    
    print("Sales Forecasting Application")
    print("="*40)
    print("\nOptions:")
    print("1. Run with demo data")
    print("2. Run with custom data")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Run with demo data
        app.run_complete_analysis(use_demo_data=True)
    
    elif choice == "2":
        # Run with custom data
        file_path = input("Enter path to your data file (CSV or Excel): ").strip()
        if os.path.exists(file_path):
            app.run_complete_analysis(use_demo_data=False, custom_data_path=file_path)
        else:
            print("File not found. Running with demo data instead...")
            app.run_complete_analysis(use_demo_data=True)
    
    else:
        print("Invalid choice. Running with demo data...")
        app.run_complete_analysis(use_demo_data=True)
    
    # Additional examples of individual functions
    print("\n" + "="*60)
    print("💡 You can also run individual components:")
    print("="*60)
    print("""
    # Initialize the app
    app = SalesForecastingApp()
    
    # Load demo data
    app.generate_demo_data()
    
    # Or load custom data
    # app.load_custom_data('your_file.csv')
    
    # Run individual steps
    app.explore_data()
    app.feature_engineering()
    app.build_models()
    app.generate_visualizations()
    app.export_results()
    """)
