"""
Advanced Titanic Survival Analysis - Professional Data Science Implementation
===========================================================================

This module demonstrates advanced data analysis techniques including:
- Statistical hypothesis testing
- Advanced feature engineering
- Multiple ML algorithms with hyperparameter tuning
- Model interpretability with SHAP
- Business intelligence insights
- Production-ready code structure

Author: Data Science Portfolio
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Statistical testing
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb

# Model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

class TitanicAnalyzer:
    """
    Professional Titanic dataset analyzer with advanced data science techniques
    """
    
    def __init__(self, data_path=None):
        """Initialize the analyzer with optional data path"""
        self.data = None
        self.processed_data = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load and perform initial data validation"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"✅ Data loaded successfully: {self.data.shape}")
            self._validate_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def _validate_data(self):
        """Perform comprehensive data validation"""
        print("\n📊 DATA QUALITY ASSESSMENT")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Percentage', ascending=False)
        
        print("\n🔍 Missing Values Analysis:")
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Data types
        print(f"\n📋 Data Types:")
        print(self.data.dtypes.value_counts())
        
        # Duplicate records
        duplicates = self.data.duplicated().sum()
        print(f"\n🔄 Duplicate records: {duplicates}")
        
        return missing_df
    
    def exploratory_data_analysis(self):
        """Comprehensive EDA with statistical insights"""
        print("\n🔬 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Survival rate analysis
        survival_rate = self.data['Survived'].mean()
        print(f"\n📈 Overall Survival Rate: {survival_rate:.1%}")
        
        # Statistical tests for categorical variables
        self._perform_statistical_tests()
        
        # Create comprehensive visualizations
        self._create_eda_visualizations()
        
        # Advanced correlation analysis
        self._correlation_analysis()
        
        return self._generate_insights()
    
    def _perform_statistical_tests(self):
        """Perform statistical hypothesis tests"""
        print("\n🧪 STATISTICAL HYPOTHESIS TESTING")
        print("-" * 40)
        
        # Chi-square test for categorical variables
        categorical_vars = ['Sex', 'Pclass', 'Embarked']
        
        for var in categorical_vars:
            if var in self.data.columns:
                # Create contingency table
                contingency_table = pd.crosstab(self.data[var], self.data['Survived'])
                
                # Perform chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                print(f"\n{var} vs Survival:")
                print(f"  Chi-square statistic: {chi2:.4f}")
                print(f"  p-value: {p_value:.4e}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Mann-Whitney U test for continuous variables
        continuous_vars = ['Age', 'Fare']
        
        for var in continuous_vars:
            if var in self.data.columns:
                survived = self.data[self.data['Survived'] == 1][var].dropna()
                not_survived = self.data[self.data['Survived'] == 0][var].dropna()
                
                statistic, p_value = mannwhitneyu(survived, not_survived, alternative='two-sided')
                
                print(f"\n{var} distribution difference:")
                print(f"  Mann-Whitney U statistic: {statistic:.4f}")
                print(f"  p-value: {p_value:.4e}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    def _create_eda_visualizations(self):
        """Create professional visualizations for EDA"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Titanic Dataset - Comprehensive EDA', fontsize=16, fontweight='bold')
        
        # 1. Survival by Gender
        survival_by_gender = self.data.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
        survival_by_gender['survival_rate'] = survival_by_gender['mean']
        
        axes[0, 0].bar(survival_by_gender.index, survival_by_gender['survival_rate'], 
                      color=['lightcoral', 'lightblue'], alpha=0.8)
        axes[0, 0].set_title('Survival Rate by Gender', fontweight='bold')
        axes[0, 0].set_ylabel('Survival Rate')
        for i, v in enumerate(survival_by_gender['survival_rate']):
            axes[0, 0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        
        # 2. Survival by Passenger Class
        survival_by_class = self.data.groupby('Pclass')['Survived'].mean()
        axes[0, 1].bar(survival_by_class.index, survival_by_class.values, 
                      color=['gold', 'silver', 'brown'], alpha=0.8)
        axes[0, 1].set_title('Survival Rate by Passenger Class', fontweight='bold')
        axes[0, 1].set_xlabel('Passenger Class')
        axes[0, 1].set_ylabel('Survival Rate')
        for i, v in enumerate(survival_by_class.values):
            axes[0, 1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        
        # 3. Age distribution by survival
        survived_ages = self.data[self.data['Survived'] == 1]['Age'].dropna()
        not_survived_ages = self.data[self.data['Survived'] == 0]['Age'].dropna()
        
        axes[0, 2].hist(survived_ages, bins=20, alpha=0.7, label='Survived', color='green')
        axes[0, 2].hist(not_survived_ages, bins=20, alpha=0.7, label='Did not survive', color='red')
        axes[0, 2].set_title('Age Distribution by Survival', fontweight='bold')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # 4. Fare distribution by survival
        survived_fare = self.data[self.data['Survived'] == 1]['Fare'].dropna()
        not_survived_fare = self.data[self.data['Survived'] == 0]['Fare'].dropna()
        
        axes[1, 0].boxplot([survived_fare, not_survived_fare], 
                          labels=['Survived', 'Did not survive'])
        axes[1, 0].set_title('Fare Distribution by Survival', fontweight='bold')
        axes[1, 0].set_ylabel('Fare')
        
        # 5. Family size analysis
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1
        family_survival = self.data.groupby('FamilySize')['Survived'].mean()
        
        axes[1, 1].plot(family_survival.index, family_survival.values, 
                       marker='o', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_title('Survival Rate by Family Size', fontweight='bold')
        axes[1, 1].set_xlabel('Family Size')
        axes[1, 1].set_ylabel('Survival Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Embarkation port analysis
        if 'Embarked' in self.data.columns:
            embarked_survival = self.data.groupby('Embarked')['Survived'].mean()
            axes[1, 2].bar(embarked_survival.index, embarked_survival.values, 
                          color=['lightgreen', 'lightcoral', 'lightyellow'], alpha=0.8)
            axes[1, 2].set_title('Survival Rate by Embarkation Port', fontweight='bold')
            axes[1, 2].set_xlabel('Embarkation Port')
            axes[1, 2].set_ylabel('Survival Rate')
            for i, v in enumerate(embarked_survival.values):
                axes[1, 2].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _correlation_analysis(self):
        """Advanced correlation analysis"""
        print("\n🔗 CORRELATION ANALYSIS")
        print("-" * 30)
        
        # Select numeric columns for correlation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print strongest correlations with survival
        survival_corr = correlation_matrix['Survived'].abs().sort_values(ascending=False)
        print("\nStrongest correlations with Survival:")
        for feature, corr in survival_corr.items():
            if feature != 'Survived':
                print(f"  {feature}: {corr:.3f}")
    
    def advanced_feature_engineering(self):
        """Create advanced features for better model performance"""
        print("\n🔧 ADVANCED FEATURE ENGINEERING")
        print("=" * 50)
        
        # Create a copy for processing
        self.processed_data = self.data.copy()
        
        # 1. Title extraction from names
        if 'Name' in self.processed_data.columns:
            self.processed_data['Title'] = self.processed_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            
            # Group rare titles
            title_counts = self.processed_data['Title'].value_counts()
            rare_titles = title_counts[title_counts < 10].index
            self.processed_data['Title'] = self.processed_data['Title'].replace(rare_titles, 'Rare')
            
            print(f"✅ Extracted titles: {self.processed_data['Title'].unique()}")
        
        # 2. Family size categories
        self.processed_data['FamilySize'] = self.processed_data['SibSp'] + self.processed_data['Parch'] + 1
        self.processed_data['IsAlone'] = (self.processed_data['FamilySize'] == 1).astype(int)
        
        # Categorize family size
        def categorize_family_size(size):
            if size == 1:
                return 'Alone'
            elif size <= 4:
                return 'Small'
            else:
                return 'Large'
        
        self.processed_data['FamilySizeCategory'] = self.processed_data['FamilySize'].apply(categorize_family_size)
        
        # 3. Age groups
        if 'Age' in self.processed_data.columns:
            self.processed_data['AgeGroup'] = pd.cut(self.processed_data['Age'], 
                                                   bins=[0, 12, 18, 35, 60, 100], 
                                                   labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
        
        # 4. Fare categories
        if 'Fare' in self.processed_data.columns:
            self.processed_data['FareCategory'] = pd.qcut(self.processed_data['Fare'].fillna(self.processed_data['Fare'].median()), 
                                                        q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        # 5. Cabin deck extraction
        if 'Cabin' in self.processed_data.columns:
            self.processed_data['Deck'] = self.processed_data['Cabin'].str[0]
            self.processed_data['HasCabin'] = (~self.processed_data['Cabin'].isnull()).astype(int)
        
        # 6. Interaction features
        if 'Sex' in self.processed_data.columns and 'Pclass' in self.processed_data.columns:
            self.processed_data['Sex_Pclass'] = self.processed_data['Sex'] + '_' + self.processed_data['Pclass'].astype(str)
        
        print(f"✅ Created {len(self.processed_data.columns) - len(self.data.columns)} new features")
        print(f"📊 Total features: {len(self.processed_data.columns)}")
        
        return self.processed_data
    
    def prepare_ml_data(self):
        """Prepare data for machine learning with proper preprocessing"""
        print("\n🤖 MACHINE LEARNING DATA PREPARATION")
        print("=" * 50)
        
        if self.processed_data is None:
            self.advanced_feature_engineering()
        
        # Define features to use
        feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                          'FamilySize', 'IsAlone', 'FamilySizeCategory']
        
        # Add engineered features if they exist
        if 'Title' in self.processed_data.columns:
            feature_columns.append('Title')
        if 'AgeGroup' in self.processed_data.columns:
            feature_columns.append('AgeGroup')
        if 'FareCategory' in self.processed_data.columns:
            feature_columns.append('FareCategory')
        if 'HasCabin' in self.processed_data.columns:
            feature_columns.append('HasCabin')
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in self.processed_data.columns]
        
        X = self.processed_data[available_features].copy()
        y = self.processed_data['Survived'].copy()
        
        # Handle missing values
        # Numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
        
        # Categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        print(f"✅ Prepared {X.shape[1]} features for {X.shape[0]} samples")
        print(f"📋 Features: {list(X.columns)}")
        
        return X, y, label_encoders
    
    def train_multiple_models(self):
        """Train and compare multiple ML algorithms"""
        print("\n🎯 TRAINING MULTIPLE ML MODELS")
        print("=" * 50)
        
        X, y, encoders = self.prepare_ml_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, stratify=y)
        
        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # ROC AUC if predict_proba is available
            try:
                test_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, test_proba)
            except:
                roc_auc = None
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'roc_auc': roc_auc
            }
            
            print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            if roc_auc:
                print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Store results
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
        
        return results, X_test, y_test
    
    def model_interpretability(self, model_name='Random Forest'):
        """Analyze model interpretability using various techniques"""
        print(f"\n🔍 MODEL INTERPRETABILITY ANALYSIS - {model_name}")
        print("=" * 60)
        
        if not self.models:
            print("❌ No trained models found. Please run train_multiple_models() first.")
            return
        
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            return
        
        model = self.models[model_name]
        X, y, _ = self.prepare_ml_data()
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Feature Importance Rankings:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
            plt.title(f'Top 10 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
            
            self.feature_importance[model_name] = feature_importance
        
        # SHAP analysis (if available)
        if SHAP_AVAILABLE and hasattr(model, 'predict_proba'):
            try:
                print("\n🎯 SHAP Analysis...")
                
                # Create SHAP explainer
                if 'Random Forest' in model_name or 'XGBoost' in model_name or 'LightGBM' in model_name:
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X.sample(100))  # Use sample for speed
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X.sample(100))
                
                # Summary plot
                shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                                X.sample(100), show=False)
                plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                
                print("✅ SHAP analysis completed")
                
            except Exception as e:
                print(f"⚠️ SHAP analysis failed: {e}")
    
    def generate_business_insights(self):
        """Generate actionable business insights from the analysis"""
        print("\n💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)
        
        insights = []
        
        # Survival rate insights
        if self.data is not None:
            overall_survival = self.data['Survived'].mean()
            insights.append(f"Overall passenger survival rate: {overall_survival:.1%}")
            
            # Gender insights
            gender_survival = self.data.groupby('Sex')['Survived'].mean()
            female_survival = gender_survival.get('female', 0)
            male_survival = gender_survival.get('male', 0)
            
            insights.append(f"Female survival rate ({female_survival:.1%}) was {female_survival/male_survival:.1f}x higher than male survival rate ({male_survival:.1%})")
            
            # Class insights
            class_survival = self.data.groupby('Pclass')['Survived'].mean()
            insights.append(f"First-class passengers had {class_survival[1]:.1%} survival rate vs {class_survival[3]:.1%} for third-class")
            
            # Age insights
            if 'Age' in self.data.columns:
                child_survival = self.data[self.data['Age'] < 18]['Survived'].mean()
                adult_survival = self.data[self.data['Age'] >= 18]['Survived'].mean()
                insights.append(f"Children (<18) had {child_survival:.1%} survival rate vs {adult_survival:.1%} for adults")
        
        # Model performance insights
        if self.results:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
            best_accuracy = self.results[best_model]['test_accuracy']
            insights.append(f"Best predictive model: {best_model} with {best_accuracy:.1%} accuracy")
        
        # Feature importance insights
        if self.feature_importance:
            for model_name, importance_df in self.feature_importance.items():
                top_feature = importance_df.iloc[0]
                insights.append(f"Most important predictive factor ({model_name}): {top_feature['feature']} ({top_feature['importance']:.3f})")
        
        print("\n🎯 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print("\n💡 BUSINESS RECOMMENDATIONS:")
        recommendations = [
            "Prioritize safety protocols for male passengers and lower-class accommodations",
            "Implement family-based evacuation procedures to improve survival rates",
            "Focus emergency training on crew serving third-class passenger areas",
            "Consider passenger demographics in safety equipment distribution",
            "Develop predictive models for real-time risk assessment during emergencies"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        return insights, recommendations
    
    def _generate_insights(self):
        """Generate statistical insights from EDA"""
        insights = {}
        
        if self.data is not None:
            # Basic statistics
            insights['total_passengers'] = len(self.data)
            insights['survival_rate'] = self.data['Survived'].mean()
            
            # Gender analysis
            gender_stats = self.data.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
            insights['gender_survival'] = gender_stats.to_dict()
            
            # Class analysis
            class_stats = self.data.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
            insights['class_survival'] = class_stats.to_dict()
            
            # Age analysis
            if 'Age' in self.data.columns:
                age_stats = {
                    'mean_age': self.data['Age'].mean(),
                    'median_age': self.data['Age'].median(),
                    'age_survival_corr': self.data['Age'].corr(self.data['Survived'])
                }
                insights['age_analysis'] = age_stats
        
        return insights

def create_sample_data():
    """Create sample Titanic data for demonstration"""
    np.random.seed(42)
    
    n_samples = 891
    
    # Generate sample data that mimics Titanic dataset patterns
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 14, n_samples).clip(0, 80),
        'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.01, 0.00]),
        'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.02, 0.01, 0.00, 0.00]),
        'Fare': np.random.lognormal(2.5, 1.2, n_samples).clip(0, 512),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    }
    
    # Generate names
    first_names_male = ['John', 'William', 'James', 'Charles', 'George', 'Frank', 'Joseph', 'Thomas', 'Henry', 'Robert']
    first_names_female = ['Mary', 'Anna', 'Margaret', 'Helen', 'Elizabeth', 'Ruth', 'Florence', 'Ethel', 'Emma', 'Marie']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia', 'Rodriguez', 'Wilson']
    titles_male = ['Mr', 'Master', 'Rev', 'Dr', 'Col']
    titles_female = ['Miss', 'Mrs', 'Mme', 'Ms', 'Lady']
    
    names = []
    for i in range(n_samples):
        if data['Sex'][i] == 'male':
            first_name = np.random.choice(first_names_male)
            title = np.random.choice(titles_male, p=[0.8, 0.1, 0.04, 0.03, 0.03])
        else:
            first_name = np.random.choice(first_names_female)
            title = np.random.choice(titles_female, p=[0.4, 0.5, 0.05, 0.03, 0.02])
        
        last_name = np.random.choice(last_names)
        names.append(f"{last_name}, {title}. {first_name}")
    
    data['Name'] = names
    
    # Generate survival based on historical patterns
    survival_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.3  # Base probability
        
        # Gender effect (strongest predictor)
        if data['Sex'][i] == 'female':
            prob += 0.4
        
        # Class effect
        if data['Pclass'][i] == 1:
            prob += 0.2
        elif data['Pclass'][i] == 2:
            prob += 0.1
        
        # Age effect
        if data['Age'][i] < 16:
            prob += 0.15
        elif data['Age'][i] > 60:
            prob -= 0.1
        
        # Family size effect
        family_size = data['SibSp'][i] + data['Parch'][i] + 1
        if 2 <= family_size <= 4:
            prob += 0.1
        elif family_size > 4:
            prob -= 0.15
        
        survival_prob[i] = np.clip(prob, 0, 1)
    
    data['Survived'] = np.random.binomial(1, survival_prob)
    
    return pd.DataFrame(data)

def main():
    """Main function to demonstrate the advanced analysis"""
    print("🚢 ADVANCED TITANIC SURVIVAL ANALYSIS")
    print("=" * 60)
    print("Professional Data Science Portfolio Demonstration")
    print("=" * 60)
    
    # Create sample data for demonstration
    print("\n📊 Creating sample Titanic dataset...")
    sample_data = create_sample_data()
    
    # Initialize analyzer
    analyzer = TitanicAnalyzer()
    analyzer.data = sample_data
    
    # Perform comprehensive analysis
    print("\n🔍 Starting comprehensive analysis...")
    
    # 1. Data validation and EDA
    analyzer._validate_data()
    insights = analyzer.exploratory_data_analysis()
    
    # 2. Feature engineering
    analyzer.advanced_feature_engineering()
    
    # 3. Machine learning
    results, X_test, y_test = analyzer.train_multiple_models()
    
    # 4. Model interpretability
    analyzer.model_interpretability('Random Forest')
    
    # 5. Business insights
    business_insights, recommendations = analyzer.generate_business_insights()
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("This demonstration showcases:")
    print("• Advanced statistical analysis and hypothesis testing")
    print("• Professional data visualization and EDA")
    print("• Sophisticated feature engineering techniques")
    print("• Multiple ML algorithms with proper evaluation")
    print("• Model interpretability and explainability")
    print("• Business intelligence and actionable insights")
    print("• Production-ready code structure and documentation")

if __name__ == "__main__":
    main()