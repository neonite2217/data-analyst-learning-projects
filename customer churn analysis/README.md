# 📊 Customer Churn Analysis - Multi-Tool Professional Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![PowerBI](https://img.shields.io/badge/Power%20BI-Ready-yellow.svg)](https://powerbi.microsoft.com/)
[![Tableau](https://img.shields.io/badge/Tableau-Compatible-green.svg)](https://www.tableau.com/)
[![Excel](https://img.shields.io/badge/Excel-Dashboard-success.svg)](https://www.microsoft.com/excel)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

> A comprehensive customer churn analysis project demonstrating professional usage of Python, Excel, Power BI, and Tableau for predictive analytics and business intelligence.

## 🎯 Project Overview

This project provides a complete end-to-end customer churn analysis solution that generates professional-grade dashboards, predictive models, and business intelligence reports. Designed to showcase intermediate to advanced data science and BI skills across multiple platforms.

### 🔥 Key Features

- **🤖 Machine Learning Pipeline**: Random Forest, Gradient Boosting, and Logistic Regression models with 85%+ accuracy
- **📊 Multi-Platform Dashboards**: Excel, Power BI, and Tableau ready datasets and templates
- **💼 Executive Reporting**: Business intelligence reports with ROI analysis and strategic recommendations
- **🔄 Multiple Datasets**: Synthetic data generation + real-world dataset integration
- **🚀 Production Ready**: Deployable model artifacts with API-ready prediction scripts
- **📈 Advanced Analytics**: Customer segmentation, lifetime value, and risk scoring

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Data Science** | Python, Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Business Intelligence** | Power BI (DAX), Tableau, Excel (Advanced Formulas) |
| **Machine Learning** | Random Forest, Gradient Boosting, Logistic Regression |
| **Data Formats** | CSV, Excel (.xlsx), Pickle (.pkl) |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Excel 2016 or higher (for dashboard viewing)
- Power BI Desktop (optional)
- Tableau Desktop (optional)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/neonite2217/data-analyst-learning-projects.git
cd "data-analyst-learning-projects/customer churn analysis"

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl requests joblib

# Run the complete analysis
python churn_analysis.py
```

### Alternative: pip install
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Run the Analysis
```bash
python churn_analysis.py
```

The script will prompt you to choose a dataset:
- **Option 1**: Synthetic Data (5,000 customers, perfect for demo)
- **Option 2**: IBM Telco Dataset (7,043 real customer records)
- **Option 3**: Banking Dataset (10,000+ bank customers)

### 2. Explore Generated Files
After running, you'll find:
```
churn_analysis_results/
├── 📊 Customer_Churn_Analysis_Dashboard.xlsx
├── data/
│   ├── powerbi_dataset.csv
│   └── tableau_dataset.csv
├── models/
│   └── churn_prediction_model.pkl
└── reports/
    ├── Business_Intelligence_Report.md
    └── Executive_Presentation.md
```

### 3. Import into BI Tools

#### Power BI
1. Open Power BI Desktop
2. Get Data → Text/CSV → Select `powerbi_dataset.csv`
3. Copy DAX measures from `reports/PowerBI_DAX_Measures.txt`
4. Follow the setup guide for dashboard creation

#### Tableau
1. Open Tableau Desktop
2. Connect to Text File → Select `tableau_dataset.csv`
3. Follow instructions in `reports/Tableau_Setup_Instructions.txt`
4. Build recommended dashboard structure

#### Excel
1. Open `Customer_Churn_Analysis_Dashboard.xlsx`
2. Enable macros if prompted
3. Explore the 6 pre-built worksheet tabs

## 📊 Generated Outputs

### 📈 Executive Dashboard (Excel)
Interactive Excel workbook with 6 comprehensive sheets:
- **Executive Summary**: Key KPIs and metrics
- **Customer Data**: Full dataset with risk scores  
- **Segment Analysis**: Churn rates by customer segments
- **High Risk Actions**: Actionable customer intervention list
- **Financial Impact**: Revenue scenarios and ROI projections
- **Cohort Analysis**: Tenure-based churn patterns

### 🔍 Business Intelligence Reports
- **Executive Presentation**: 9-slide strategic overview with recommendations
- **Technical Report**: Detailed analysis methodology and findings
- **ROI Analysis**: Financial impact and investment recommendations

### 🤖 Machine Learning Assets
- **Trained Models**: Production-ready pickle files
- **Deployment Script**: API-ready prediction class
- **Feature Engineering**: Reusable preprocessing pipeline
- **Model Performance**: Detailed evaluation metrics

### 📊 BI Platform Assets
- **Power BI**: 25+ DAX measures + setup guide
- **Tableau**: Calculated fields + dashboard templates  
- **Excel**: Advanced formulas + pivot table structures

## 🎯 Business Use Cases

### 📞 Telecommunications
- Customer retention strategies
- Contract optimization analysis
- Service bundle recommendations
- Support ticket impact assessment

### 🏦 Financial Services  
- Account closure prediction
- Customer lifetime value optimization
- Risk-based pricing strategies
- Cross-selling opportunity identification

### 🛒 E-commerce/SaaS
- Subscription churn prevention
- User engagement optimization
- Pricing strategy development
- Customer success program design

## 📊 Sample Results & KPIs

| Metric | Value | Impact |
|--------|--------|---------|
| **Model Accuracy** | 85-90% AUC | High confidence predictions |
| **Customers Analyzed** | 5,000+ | Comprehensive dataset |
| **High-Risk Identified** | ~15% of base | Focused retention efforts |
| **Revenue at Risk** | $156K+ monthly | Clear financial impact |
| **Potential ROI** | 300-500% | Strong business case |
| **Churn Reduction Target** | 20-30% | Achievable improvement |

## 🔧 Advanced Usage

### Custom Dataset Integration
```python
# Load your own dataset
project = ChurnAnalysisProject()
project.df = pd.read_csv('your_custom_dataset.csv')
project.engineer_features()
project.train_models()
project.create_excel_dashboard()
```

### Model Deployment
```python
# Load trained model for predictions
from models.model_deployment import ChurnPredictor

predictor = ChurnPredictor(
    model_path='models/churn_prediction_model.pkl',
    scaler_path='models/feature_scaler.pkl',
    features_path='models/feature_columns.json'
)

# Predict churn for new customer
result = predictor.predict_churn({
    'Age': 45,
    'Monthly_Charges': 75.50,
    'Tenure_Months': 12,
    'Contract_Encoded': 0
})

print(f"Churn Probability: {result['churn_probability'][0]:.3f}")
print(f"Risk Level: {result['risk_level'][0]}")
```

### API Integration
The deployment script is ready for Flask/FastAPI integration:
```python
from flask import Flask, request, jsonify
from models.model_deployment import ChurnPredictor

app = Flask(__name__)
predictor = ChurnPredictor(...)

@app.route('/predict', methods=['POST'])
def predict():
    customer_data = request.json
    result = predictor.predict_churn(customer_data)
    return jsonify(result)
```

## 📚 Project Structure

```
data-analyst-learning-projects/
│
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 🐍 churn_analysis.py                  # Main analysis script
│
├── 📊 churn_analysis_results/            # Generated outputs
│   ├── 📊 Customer_Churn_Analysis_Dashboard.xlsx
│   │
│   ├── 📁 data/                          # Datasets
│   │   ├── raw_customer_data.csv
│   │   ├── powerbi_dataset.csv
│   │   └── tableau_dataset.csv
│   │
│   ├── 📁 models/                        # ML artifacts  
│   │   ├── churn_prediction_model.pkl
│   │   ├── feature_scaler.pkl
│   │   ├── feature_columns.json
│   │   └── model_deployment.py
│   │
│   ├── 📁 visualizations/                # Charts & graphs
│   │   ├── comprehensive_eda.png
│   │   └── feature_importance.png
│   │
│   └── 📁 reports/                       # Business reports
│       ├── Business_Intelligence_Report.md
│       ├── Executive_Presentation.md
│       ├── PowerBI_DAX_Measures.txt
│       ├── PowerBI_Setup_Guide.txt
│       └── Tableau_Setup_Instructions.txt
│
└── 📁 examples/                          # Usage examples
    ├── api_integration_example.py
    ├── custom_dataset_example.py
    └── dashboard_customization_guide.md
```

## 🎓 Learning Outcomes

### Data Science Skills
- ✅ End-to-end ML pipeline development
- ✅ Feature engineering and selection
- ✅ Model evaluation and comparison
- ✅ Production deployment preparation

### Business Intelligence
- ✅ Cross-platform dashboard development
- ✅ KPI definition and metric calculation
- ✅ Executive communication and reporting
- ✅ ROI analysis and business case development

### Technical Proficiency
- ✅ Python data analysis ecosystem
- ✅ Advanced Excel formula and pivot table usage
- ✅ Power BI DAX measure creation
- ✅ Tableau calculated field development

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Bug Reports**: Open an issue describing the problem
2. **💡 Feature Requests**: Suggest new functionality or improvements
3. **🔧 Code Contributions**: Fork, develop, and submit a pull request
4. **📚 Documentation**: Improve guides and examples

### Development Setup
```bash
# Fork and clone the repo
git clone https://github.com/neonite2217/data-analyst-learning-projects.git
cd "data-analyst-learning-projects/customer churn analysis"

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python churn_analysis.py

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature-name

# Submit a pull request
```

## 🎯 Roadmap

### Version 2.0 Planned Features
- [ ] **Real-time Streaming**: Kafka + Spark integration
- [ ] **Advanced ML**: Deep learning models (TensorFlow/PyTorch)
- [ ] **AutoML**: Automated model selection and hyperparameter tuning
- [ ] **MLOps**: Model versioning and A/B testing framework
- [ ] **Web App**: Streamlit/Dash interactive dashboard
- [ ] **Cloud Deployment**: AWS/Azure/GCP deployment guides

### Integration Enhancements
- [ ] **CRM Integration**: Salesforce/HubSpot connectors
- [ ] **Marketing Automation**: Campaign trigger systems
- [ ] **Customer Support**: Zendesk/ServiceNow workflows
- [ ] **Financial Systems**: ERP integration capabilities

## 📖 Documentation

### 📚 Detailed Guides
- [Power BI Setup Guide](churn_analysis_results/reports/PowerBI_Setup_Guide.txt)
- [Tableau Instructions](churn_analysis_results/reports/Tableau_Setup_Instructions.txt)
- [Model Deployment Guide](churn_analysis_results/models/model_deployment.py)
- [Business Report Template](churn_analysis_results/reports/Business_Intelligence_Report.md)

### 🎥 Video Tutorials (Coming Soon)
- End-to-end project walkthrough
- Power BI dashboard creation
- Tableau visualization building  
- Excel advanced features demo

## ❓ FAQ

**Q: Can I use my own dataset?**  
A: Yes! The script supports custom CSV imports. Ensure your data has customer ID, churn indicator, and relevant features.

**Q: What if I don't have Power BI/Tableau?**  
A: The Excel dashboard provides comprehensive analysis. Power BI Desktop is free, and Tableau offers student licenses.

**Q: How accurate are the predictions?**  
A: The models achieve 85-90% AUC scores on test data. Real-world performance depends on data quality and business context.

**Q: Can this be used for other industries?**  
A: Absolutely! While optimized for telecom, the framework applies to any subscription-based business (SaaS, banking, insurance, etc.).

**Q: Is this suitable for production use?**  
A: The models and deployment scripts are production-ready. However, consider additional validation, monitoring, and compliance requirements for your specific use case.

## 🏆 Showcase & Portfolio Use

This project demonstrates:

### 🎯 **For Data Scientists**
- Complete ML pipeline with multiple algorithms
- Feature engineering and model evaluation expertise
- Production deployment readiness
- Business acumen and ROI focus

### 📊 **For BI Analysts**  
- Cross-platform dashboard development
- Advanced Excel, Power BI, and Tableau skills
- Executive communication abilities
- Strategic business recommendations

### 💼 **For Business Analysts**
- Customer analytics and segmentation
- Financial impact quantification  
- Actionable insight generation
- Stakeholder presentation skills

### 🔧 **For Data Engineers**
- Data pipeline development
- Model deployment architecture
- API integration capabilities
- Scalable system design

## 📞 Support & Contact

### 🆘 Getting Help
- **Issues**: [GitHub Issues](https://github.com/neonite2217/data-analyst-learning-projects/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neonite2217/data-analyst-learning-projects/discussions)
- **Email**: [your.email@domain.com](mailto:your.email@domain.com)

### 🌐 Connect
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **Portfolio**: [Your Portfolio Website](https://yourportfolio.com)
- **Blog**: [Your Data Science Blog](https://yourblog.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM**: For the Telco Customer Churn dataset
- **Scikit-learn**: For the machine learning algorithms
- **Pandas**: For data manipulation capabilities
- **Power BI Community**: For DAX formula inspiration
- **Tableau Community**: For visualization best practices

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neonite2217/data-analyst-learning-projects&type=Date)](https://star-history.com/#neonite2217/data-analyst-learning-projects&Date)

---

### 🚀 Ready to Get Started?

```bash
git clone https://github.com/neonite2217/data-analyst-learning-projects.git
cd "data-analyst-learning-projects/customer churn analysis"
pip install -r requirements.txt
python churn_analysis.py
```

**Happy Analyzing! 📊✨**

---

> *"In God we trust. All others must bring data."* - W. Edwards Deming

[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/neonite2217)
[![Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![Data Science](https://img.shields.io/badge/Powered%20by-Data%20Science-green.svg)](https://github.com/neonite2217/data-analyst-learning-projects)