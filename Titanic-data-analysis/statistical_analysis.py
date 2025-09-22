"""
Advanced Statistical Analysis for Titanic Dataset
===============================================

This module demonstrates advanced statistical techniques including:
- Hypothesis testing (Chi-square, t-tests, ANOVA)
- Survival analysis with Kaplan-Meier curves
- Bayesian analysis
- Time series analysis (if applicable)
- Advanced statistical modeling
- Confidence intervals and effect sizes

Author: Data Science Portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, mannwhitneyu
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit

# Survival analysis
try:
    from lifelines import KaplanMeierFitter, LogRankTest
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Lifelines not available. Install with: pip install lifelines")

class AdvancedStatisticalAnalyzer:
    """
    Advanced statistical analysis class for Titanic dataset
    """
    
    def __init__(self, data):
        """Initialize with dataset"""
        self.data = data.copy()
        self.results = {}
        self.statistical_tests = {}
        
    def comprehensive_hypothesis_testing(self):
        """Perform comprehensive hypothesis testing suite"""
        print("🧪 COMPREHENSIVE HYPOTHESIS TESTING SUITE")
        print("=" * 60)
        
        results = {}
        
        # 1. Chi-square tests for categorical associations
        print("\n1️⃣ CHI-SQUARE TESTS FOR INDEPENDENCE")
        print("-" * 40)
        
        categorical_vars = ['Sex', 'Pclass', 'Embarked']
        
        for var in categorical_vars:
            if var in self.data.columns:
                # Create contingency table
                contingency_table = pd.crosstab(self.data[var], self.data['Survived'])
                
                # Perform chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Calculate effect size (Cramér's V)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                
                results[f'chi2_{var}'] = {
                    'statistic': chi2,
                    'p_value': p_value,
                    'dof': dof,
                    'cramers_v': cramers_v,
                    'significant': p_value < 0.05
                }
                
                print(f"\n{var} vs Survival:")
                print(f"  χ² = {chi2:.4f}, p = {p_value:.4e}, df = {dof}")
                print(f"  Cramér's V = {cramers_v:.4f} (Effect size)")
                print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'}")
                
                # Display contingency table with percentages
                print(f"  Contingency Table:")
                ct_percent = pd.crosstab(self.data[var], self.data['Survived'], normalize='index') * 100
                print(ct_percent.round(1))
        
        # 2. T-tests and Mann-Whitney U tests for continuous variables
        print("\n\n2️⃣ TESTS FOR CONTINUOUS VARIABLES")
        print("-" * 40)
        
        continuous_vars = ['Age', 'Fare']
        
        for var in continuous_vars:
            if var in self.data.columns:
                survived = self.data[self.data['Survived'] == 1][var].dropna()
                not_survived = self.data[self.data['Survived'] == 0][var].dropna()
                
                # Normality tests
                _, p_norm_surv = stats.shapiro(survived.sample(min(5000, len(survived))))
                _, p_norm_not = stats.shapiro(not_survived.sample(min(5000, len(not_survived))))
                
                # Choose appropriate test based on normality
                if p_norm_surv > 0.05 and p_norm_not > 0.05:
                    # Use t-test if both groups are normal
                    t_stat, p_value = ttest_ind(survived, not_survived)
                    test_name = "Independent t-test"
                    
                    # Calculate Cohen's d (effect size)
                    pooled_std = np.sqrt(((len(survived) - 1) * survived.var() + 
                                        (len(not_survived) - 1) * not_survived.var()) / 
                                       (len(survived) + len(not_survived) - 2))
                    cohens_d = (survived.mean() - not_survived.mean()) / pooled_std
                    effect_size = cohens_d
                    
                else:
                    # Use Mann-Whitney U test if not normal
                    t_stat, p_value = mannwhitneyu(survived, not_survived, alternative='two-sided')
                    test_name = "Mann-Whitney U test"
                    
                    # Calculate rank-biserial correlation (effect size for Mann-Whitney)
                    n1, n2 = len(survived), len(not_survived)
                    effect_size = 1 - (2 * t_stat) / (n1 * n2)
                
                results[f'continuous_{var}'] = {
                    'test': test_name,
                    'statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05,
                    'survived_mean': survived.mean(),
                    'not_survived_mean': not_survived.mean()
                }
                
                print(f"\n{var} Analysis:")
                print(f"  Test: {test_name}")
                print(f"  Statistic = {t_stat:.4f}, p = {p_value:.4e}")
                print(f"  Effect size = {effect_size:.4f}")
                print(f"  Survived mean: {survived.mean():.2f}")
                print(f"  Not survived mean: {not_survived.mean():.2f}")
                print(f"  Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")
        
        # 3. ANOVA for multi-group comparisons
        print("\n\n3️⃣ ANALYSIS OF VARIANCE (ANOVA)")
        print("-" * 40)
        
        # Age by passenger class
        if 'Age' in self.data.columns and 'Pclass' in self.data.columns:
            age_by_class = [self.data[self.data['Pclass'] == cls]['Age'].dropna() 
                           for cls in [1, 2, 3]]
            
            f_stat, p_value = f_oneway(*age_by_class)
            
            # Calculate eta-squared (effect size for ANOVA)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(age_by_class)))**2 
                           for group in age_by_class)
            ss_total = sum((np.concatenate(age_by_class) - np.mean(np.concatenate(age_by_class)))**2)
            eta_squared = ss_between / ss_total
            
            results['anova_age_class'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': p_value < 0.05
            }
            
            print(f"\nAge by Passenger Class (ANOVA):")
            print(f"  F = {f_stat:.4f}, p = {p_value:.4e}")
            print(f"  η² = {eta_squared:.4f} (Effect size)")
            print(f"  Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")
        
        self.statistical_tests = results
        return results
    
    def correlation_analysis(self):
        """Advanced correlation analysis with multiple methods"""
        print("\n\n🔗 ADVANCED CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Select numeric variables
        numeric_vars = self.data.select_dtypes(include=[np.number]).columns
        numeric_data = self.data[numeric_vars]
        
        correlations = {}
        
        # 1. Pearson correlation
        pearson_corr = numeric_data.corr(method='pearson')
        
        # 2. Spearman correlation (rank-based)
        spearman_corr = numeric_data.corr(method='spearman')
        
        # 3. Kendall's tau correlation
        kendall_corr = numeric_data.corr(method='kendall')
        
        print("Correlation with Survival (Pearson | Spearman | Kendall):")
        print("-" * 55)
        
        for var in numeric_vars:
            if var != 'Survived':
                pearson_val = pearson_corr.loc['Survived', var]
                spearman_val = spearman_corr.loc['Survived', var]
                kendall_val = kendall_corr.loc['Survived', var]
                
                # Calculate p-values
                _, p_pearson = pearsonr(self.data['Survived'], self.data[var].fillna(0))
                _, p_spearman = spearmanr(self.data['Survived'], self.data[var].fillna(0))
                _, p_kendall = kendalltau(self.data['Survived'], self.data[var].fillna(0))
                
                correlations[var] = {
                    'pearson': {'r': pearson_val, 'p': p_pearson},
                    'spearman': {'r': spearman_val, 'p': p_spearman},
                    'kendall': {'r': kendall_val, 'p': p_kendall}
                }
                
                print(f"{var:12}: {pearson_val:6.3f} | {spearman_val:8.3f} | {kendall_val:7.3f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Pearson correlation heatmap
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('Pearson Correlation', fontweight='bold')
        
        # Spearman correlation heatmap
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('Spearman Correlation', fontweight='bold')
        
        # Kendall correlation heatmap
        sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
        axes[2].set_title('Kendall Correlation', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def survival_analysis(self):
        """Perform survival analysis using Kaplan-Meier estimation"""
        if not LIFELINES_AVAILABLE:
            print("⚠️ Survival analysis requires lifelines library")
            return None
        
        print("\n\n⏱️ SURVIVAL ANALYSIS")
        print("=" * 30)
        
        # For Titanic, we'll treat 'Survived' as the event and create artificial time data
        # In real survival analysis, you'd have actual time-to-event data
        
        # Create artificial time data (for demonstration)
        np.random.seed(42)
        self.data['time_to_event'] = np.random.exponential(scale=2, size=len(self.data))
        
        # Kaplan-Meier estimation
        kmf = KaplanMeierFitter()
        
        # Overall survival curve
        kmf.fit(self.data['time_to_event'], self.data['Survived'], label='Overall')
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Overall survival curve
        plt.subplot(2, 2, 1)
        kmf.plot_survival_function()
        plt.title('Overall Survival Curve', fontweight='bold')
        plt.ylabel('Survival Probability')
        plt.xlabel('Time')
        
        # Plot 2: Survival by gender
        plt.subplot(2, 2, 2)
        for gender in ['male', 'female']:
            mask = self.data['Sex'] == gender
            kmf.fit(self.data.loc[mask, 'time_to_event'], 
                   self.data.loc[mask, 'Survived'], 
                   label=gender.capitalize())
            kmf.plot_survival_function()
        
        plt.title('Survival by Gender', fontweight='bold')
        plt.ylabel('Survival Probability')
        plt.xlabel('Time')
        plt.legend()
        
        # Plot 3: Survival by passenger class
        plt.subplot(2, 2, 3)
        for pclass in [1, 2, 3]:
            mask = self.data['Pclass'] == pclass
            kmf.fit(self.data.loc[mask, 'time_to_event'], 
                   self.data.loc[mask, 'Survived'], 
                   label=f'Class {pclass}')
            kmf.plot_survival_function()
        
        plt.title('Survival by Passenger Class', fontweight='bold')
        plt.ylabel('Survival Probability')
        plt.xlabel('Time')
        plt.legend()
        
        # Plot 4: Log-rank test results
        plt.subplot(2, 2, 4)
        
        # Perform log-rank tests
        male_data = self.data[self.data['Sex'] == 'male']
        female_data = self.data[self.data['Sex'] == 'female']
        
        results = logrank_test(male_data['time_to_event'], female_data['time_to_event'],
                              male_data['Survived'], female_data['Survived'])
        
        # Create a text plot showing test results
        plt.text(0.1, 0.8, f"Log-Rank Test (Gender)", fontsize=14, fontweight='bold')
        plt.text(0.1, 0.6, f"Test Statistic: {results.test_statistic:.4f}", fontsize=12)
        plt.text(0.1, 0.4, f"p-value: {results.p_value:.4e}", fontsize=12)
        plt.text(0.1, 0.2, f"Significant: {'Yes' if results.p_value < 0.05 else 'No'}", 
                fontsize=12, color='red' if results.p_value < 0.05 else 'green')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def bayesian_analysis(self):
        """Perform Bayesian analysis for survival probability"""
        print("\n\n🎯 BAYESIAN ANALYSIS")
        print("=" * 30)
        
        # Prior beliefs about survival probability
        # Using Beta distribution as conjugate prior for Binomial likelihood
        
        # Prior parameters (weakly informative prior)
        alpha_prior = 1  # Prior successes
        beta_prior = 1   # Prior failures
        
        # Observed data
        survivors = self.data['Survived'].sum()
        total = len(self.data)
        deaths = total - survivors
        
        # Posterior parameters (Beta distribution)
        alpha_posterior = alpha_prior + survivors
        beta_posterior = beta_prior + deaths
        
        # Calculate posterior statistics
        posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
        posterior_var = (alpha_posterior * beta_posterior) / \
                       ((alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1))
        posterior_std = np.sqrt(posterior_var)
        
        # Credible interval (Bayesian confidence interval)
        from scipy.stats import beta
        credible_interval = beta.interval(0.95, alpha_posterior, beta_posterior)
        
        print(f"Bayesian Analysis Results:")
        print(f"  Prior: Beta({alpha_prior}, {beta_prior})")
        print(f"  Observed: {survivors} survivors out of {total} passengers")
        print(f"  Posterior: Beta({alpha_posterior}, {beta_posterior})")
        print(f"  Posterior Mean: {posterior_mean:.4f}")
        print(f"  Posterior Std: {posterior_std:.4f}")
        print(f"  95% Credible Interval: ({credible_interval[0]:.4f}, {credible_interval[1]:.4f})")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Prior vs Posterior
        x = np.linspace(0, 1, 1000)
        prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
        posterior_pdf = beta.pdf(x, alpha_posterior, beta_posterior)
        
        axes[0].plot(x, prior_pdf, label='Prior', linestyle='--', alpha=0.7)
        axes[0].plot(x, posterior_pdf, label='Posterior', linewidth=2)
        axes[0].axvline(posterior_mean, color='red', linestyle=':', label='Posterior Mean')
        axes[0].fill_between(x, 0, posterior_pdf, 
                           where=(x >= credible_interval[0]) & (x <= credible_interval[1]),
                           alpha=0.3, label='95% Credible Interval')
        axes[0].set_xlabel('Survival Probability')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Prior vs Posterior Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Bayesian updating process
        sample_sizes = np.arange(10, total + 1, 10)
        posterior_means = []
        
        for n in sample_sizes:
            sample_survivors = self.data['Survived'].iloc[:n].sum()
            sample_deaths = n - sample_survivors
            alpha_n = alpha_prior + sample_survivors
            beta_n = beta_prior + sample_deaths
            posterior_means.append(alpha_n / (alpha_n + beta_n))
        
        axes[1].plot(sample_sizes, posterior_means, marker='o', markersize=4)
        axes[1].axhline(survivors / total, color='red', linestyle='--', 
                       label='True Proportion')
        axes[1].set_xlabel('Sample Size')
        axes[1].set_ylabel('Posterior Mean')
        axes[1].set_title('Bayesian Learning Process', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval': credible_interval,
            'alpha_posterior': alpha_posterior,
            'beta_posterior': beta_posterior
        }
    
    def advanced_regression_analysis(self):
        """Perform advanced regression analysis"""
        print("\n\n📈 ADVANCED REGRESSION ANALYSIS")
        print("=" * 40)
        
        # Prepare data for regression
        reg_data = self.data.copy()
        
        # Handle missing values
        reg_data['Age'].fillna(reg_data['Age'].median(), inplace=True)
        reg_data['Fare'].fillna(reg_data['Fare'].median(), inplace=True)
        reg_data['Embarked'].fillna(reg_data['Embarked'].mode()[0], inplace=True)
        
        # Create dummy variables
        reg_data = pd.get_dummies(reg_data, columns=['Sex', 'Embarked'], drop_first=True)
        
        # 1. Logistic Regression with statsmodels
        print("1️⃣ LOGISTIC REGRESSION ANALYSIS")
        print("-" * 35)
        
        # Define formula
        formula = 'Survived ~ Pclass + Age + SibSp + Parch + Fare + Sex_male + Embarked_Q + Embarked_S'
        
        # Fit logistic regression
        logit_model = logit(formula, data=reg_data).fit()
        
        print(logit_model.summary())
        
        # Calculate odds ratios
        odds_ratios = np.exp(logit_model.params)
        conf_int = np.exp(logit_model.conf_int())
        
        print("\nOdds Ratios and 95% Confidence Intervals:")
        print("-" * 45)
        for var in logit_model.params.index:
            if var != 'Intercept':
                or_val = odds_ratios[var]
                ci_lower = conf_int.loc[var, 0]
                ci_upper = conf_int.loc[var, 1]
                print(f"{var:15}: OR = {or_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # 2. Model diagnostics
        print("\n\n2️⃣ MODEL DIAGNOSTICS")
        print("-" * 25)
        
        # Pseudo R-squared
        null_deviance = logit_model.null_deviance
        model_deviance = logit_model.deviance
        pseudo_r2 = 1 - (model_deviance / null_deviance)
        
        print(f"McFadden's Pseudo R²: {pseudo_r2:.4f}")
        print(f"AIC: {logit_model.aic:.2f}")
        print(f"BIC: {logit_model.bic:.2f}")
        print(f"Log-Likelihood: {logit_model.llf:.2f}")
        
        # Hosmer-Lemeshow test (goodness of fit)
        predicted_probs = logit_model.predict()
        
        # Create deciles for Hosmer-Lemeshow test
        deciles = pd.qcut(predicted_probs, 10, duplicates='drop')
        hl_table = pd.crosstab(deciles, reg_data['Survived'])
        
        print(f"\nHosmer-Lemeshow Goodness of Fit:")
        print(hl_table)
        
        # 3. Visualization of results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Coefficients plot
        coef_df = pd.DataFrame({
            'Variable': logit_model.params.index[1:],  # Exclude intercept
            'Coefficient': logit_model.params.values[1:],
            'Std_Error': logit_model.bse.values[1:]
        })
        
        axes[0, 0].barh(coef_df['Variable'], coef_df['Coefficient'], 
                       xerr=coef_df['Std_Error'], capsize=5)
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Coefficient Value')
        axes[0, 0].set_title('Logistic Regression Coefficients', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Odds ratios plot
        or_df = pd.DataFrame({
            'Variable': odds_ratios.index[1:],  # Exclude intercept
            'Odds_Ratio': odds_ratios.values[1:],
            'CI_Lower': conf_int.iloc[1:, 0].values,
            'CI_Upper': conf_int.iloc[1:, 1].values
        })
        
        axes[0, 1].scatter(or_df['Odds_Ratio'], range(len(or_df)), s=50)
        for i, (lower, upper) in enumerate(zip(or_df['CI_Lower'], or_df['CI_Upper'])):
            axes[0, 1].plot([lower, upper], [i, i], 'b-', alpha=0.7)
        
        axes[0, 1].axvline(1, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Odds Ratio')
        axes[0, 1].set_yticks(range(len(or_df)))
        axes[0, 1].set_yticklabels(or_df['Variable'])
        axes[0, 1].set_title('Odds Ratios with 95% CI', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predicted probabilities vs actual
        axes[1, 0].scatter(predicted_probs, reg_data['Survived'], alpha=0.5)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.7)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Actual Survival')
        axes[1, 0].set_title('Predicted vs Actual', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = logit_model.resid_pearson
        axes[1, 1].scatter(predicted_probs, residuals, alpha=0.5)
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Pearson Residuals')
        axes[1, 1].set_title('Residuals vs Fitted', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return logit_model, odds_ratios
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical report"""
        print("\n\n📊 COMPREHENSIVE STATISTICAL REPORT")
        print("=" * 60)
        
        report = {
            'dataset_summary': {
                'total_passengers': len(self.data),
                'survival_rate': self.data['Survived'].mean(),
                'missing_data': self.data.isnull().sum().to_dict()
            }
        }
        
        # Run all analyses
        hypothesis_results = self.comprehensive_hypothesis_testing()
        correlation_results = self.correlation_analysis()
        
        if LIFELINES_AVAILABLE:
            survival_results = self.survival_analysis()
            report['survival_analysis'] = survival_results
        
        bayesian_results = self.bayesian_analysis()
        regression_model, odds_ratios = self.advanced_regression_analysis()
        
        # Compile results
        report.update({
            'hypothesis_tests': hypothesis_results,
            'correlations': correlation_results,
            'bayesian_analysis': bayesian_results,
            'regression_summary': {
                'pseudo_r2': 1 - (regression_model.deviance / regression_model.null_deviance),
                'aic': regression_model.aic,
                'bic': regression_model.bic,
                'odds_ratios': odds_ratios.to_dict()
            }
        })
        
        # Summary insights
        print("\n🎯 KEY STATISTICAL INSIGHTS:")
        print("-" * 30)
        
        insights = []
        
        # Hypothesis testing insights
        significant_vars = [var for var, result in hypothesis_results.items() 
                          if result.get('significant', False)]
        insights.append(f"Significant associations found: {len(significant_vars)} variables")
        
        # Correlation insights
        if correlation_results:
            strongest_corr = max(correlation_results.items(), 
                               key=lambda x: abs(x[1]['pearson']['r']))
            insights.append(f"Strongest correlation with survival: {strongest_corr[0]} (r = {strongest_corr[1]['pearson']['r']:.3f})")
        
        # Bayesian insights
        ci_width = bayesian_results['credible_interval'][1] - bayesian_results['credible_interval'][0]
        insights.append(f"Survival probability 95% credible interval width: {ci_width:.3f}")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return report

def main():
    """Main function to demonstrate advanced statistical analysis"""
    print("📊 ADVANCED STATISTICAL ANALYSIS - TITANIC DATASET")
    print("=" * 70)
    
    # Create sample data
    from advanced_titanic_analysis import create_sample_data
    data = create_sample_data()
    
    # Initialize analyzer
    analyzer = AdvancedStatisticalAnalyzer(data)
    
    # Generate comprehensive report
    report = analyzer.generate_statistical_report()
    
    print("\n✅ STATISTICAL ANALYSIS COMPLETE!")
    print("=" * 50)
    print("This analysis demonstrates:")
    print("• Comprehensive hypothesis testing (Chi-square, t-tests, ANOVA)")
    print("• Advanced correlation analysis (Pearson, Spearman, Kendall)")
    print("• Survival analysis with Kaplan-Meier curves")
    print("• Bayesian statistical inference")
    print("• Advanced logistic regression with diagnostics")
    print("• Effect size calculations and confidence intervals")
    print("• Professional statistical reporting")

if __name__ == "__main__":
    main()