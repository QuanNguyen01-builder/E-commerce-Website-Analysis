# E-Commerce Customer Analytics Project

This repository contains R code and resources for analyzing and modeling customer behavior in an e-commerce setting.  
The goal is to generate actionable insights for marketing, cross-selling, and customer retention using a real-world dataset.

## 📊 Data Source

- **Dataset**: [Kaggle E-Commerce Data][https://www.kaggle.com/datasets/rishikumarrajvansh/marketing-insights-for-e-commerce-company/data]  
  The data contains transaction-level sales, customer demographics, product details, discount usage, marketing spend, and tax rates.

> *Note: The original data files used for analysis are not included in this repo due to copyright. Please download them directly from Kaggle and place them in your working directory.*

## 🔍 Project Objectives

- **Customer Segmentation** using RFM analysis and k-means clustering.
- **Customer Lifetime Value (CLV) Prediction** with logistic regression, random forest, and XGBoost.
- **Market Basket Analysis** to identify cross-sell opportunities via association rules mining (Apriori algorithm).
- **Next Purchase Prediction** to support customer retention and campaign timing.

## 🛠️ Methods and Tools

- Data cleaning and wrangling: `dplyr`, `readxl`
- Segmentation & clustering: `dplyr`, `kmeans`
- Predictive modeling: `caret`, `randomForest`, `xgboost`, `nnet`
- Market basket analysis: `arules`, `arulesViz`
- Visualization: `ggplot2`

## 🚀 Getting Started

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce-analytics.git
   cd ecommerce-analytics
