# Air Quality Alert System

A machine learning project that predicts air quality alert levels and provides health research insights to help individuals with respiratory conditions plan their outdoor activities safely.

## Project Overview

This project builds a predictive alert system using air quality data from 6 major cities. The system combines EPA AQI standards with WHO PM2.5 guidelines to create a three-tier alert system:

- **All Clear**: Safe air quality conditions
- **Caution**: Elevated pollution levels, sensitive groups should take care
- **Warning**: Unhealthy conditions, limit outdoor exposure

The project also integrates academic research on air pollution health impacts, enabling users to explore relevant scientific findings through a semantic search interface.

## Features

- **Predictive Modeling**: Machine learning models achieving 97% accuracy in predicting air quality alert levels
- **Time Series Forecasting**: ARIMA-based forecasting for next-day AQI predictions
- **Research Database**: 312 academic papers on air pollution health effects from PubMed
- **Semantic Search**: Ask questions about air quality and health to find relevant research papers
- **Interactive Dashboard**: Streamlit web application for exploring air quality insights

## Data

- **Source**: Hourly air quality measurements from 2025
- **Cities**: Brasilia, Cairo, Dubai, London, New York, Sydney
- **Features**: CO, NO2, SO2, O3, PM2.5, PM10, AQI
- **Records**: 52,560 hourly observations

## Key Findings

- **PM2.5 is the strongest predictor** of air quality alerts, consistent with WHO guidance that it's the most dangerous pollutant with no safe threshold
- **City location matters**: Including city as a feature improved model performance from 93% to 97% accuracy
- **Class imbalance varies by city**: Dubai has very few "All Clear" days (only 2 out of 365), making city-specific forecasting challenging

## Models

| Model | Target | Accuracy | Notes |
|-------|--------|----------|-------|
| Logistic Regression | AQI Category | 87% | Baseline |
| Logistic Regression | Alert Level | 93% | AQI + PM2.5 combined |
| Random Forest | Alert Level | 93% | Same as logistic regression |
| **Random Forest + Cities** | **Alert Level** | **97%** | **Best model** |
| Time Series (Dubai) | Tomorrow's AQI | 68% | 100% recall on Unhealthy days |

## Project Structure

```
├── app.py                      # Streamlit web app with semantic search
├── AQI_datasets/               # Raw hourly air quality data
│   ├── Air_Quality.csv         # Consolidated data from all cities
│   └── [City]_Air_Quality.csv  # Individual city files
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis & ML models
│   ├── api.ipynb               # PubMed API integration for research papers
│   ├── papers.csv              # Database of 312 research papers
│   ├── embeddings.npy          # Pre-computed embeddings for semantic search
│   └── processed_aqi_data.csv  # Enhanced data with alert categories
├── images/                     # Visualizations and infographics
└── README.md
```

## Web Application

The Streamlit app provides:

- Information dashboard on air pollution health effects
- PM2.5 health impact visualizations
- Semantic search functionality to query research papers
- Returns top 3 most relevant papers based on your question

### Running the App

```bash
streamlit run app.py
```

## Business Context

**Problem**: People with allergies, asthma, or other respiratory conditions need advance warning about poor air quality days.

**Solution**: A predictive alert system that prioritizes **recall over precision** — it's better to warn users unnecessarily than to miss an unhealthy day (false negatives are worse than false positives).

## Research Integration

The project fetches and indexes academic papers from PubMed covering four key health topics:

- Air pollution and lung health
- PM2.5 and respiratory disease
- Particulate matter and asthma
- AQI and cancer

Users can ask natural language questions through the app to find relevant research findings.

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- statsmodels
- streamlit
- sentence-transformers

### Installation

```bash
pip install pandas numpy scikit-learn seaborn matplotlib statsmodels streamlit sentence-transformers
```

## Next Steps

- [ ] Deploy best model as an API endpoint
- [ ] Add more cities and historical data
- [ ] Incorporate weather data for improved forecasting
- [ ] Build mobile notifications for air quality alerts
