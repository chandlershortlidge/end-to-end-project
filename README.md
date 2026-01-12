# Air Quality Alert System

A machine learning project that predicts air quality alert levels to help health-conscious individuals plan their outdoor activities.

## Project Overview

This project builds a predictive alert system using air quality data from 6 major cities. The system combines EPA AQI standards with WHO PM2.5 guidelines to create a three-tier alert system:

- **All Clear**: Safe air quality conditions
- **Caution**: Elevated pollution levels, sensitive groups should take care
- **Warning**: Unhealthy conditions, limit outdoor exposure

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
├── notebooks/
│   └── eda.ipynb          # Main analysis and modeling notebook
├── AQI_datasets/
│   └── Air_Quality.csv    # Raw air quality data
└── README.md
```

## Business Context

**Problem**: People with allergies, asthma, or other respiratory conditions need advance warning about poor air quality days.

**Solution**: A predictive alert system that prioritizes **recall over precision** — it's better to warn users unnecessarily than to miss an unhealthy day (false negatives are worse than false positives).

## Next Steps

- [ ] Deploy best model as an API
- [ ] Add more cities and historical data
- [ ] Incorporate weather data for improved forecasting
- [ ] Build user-facing dashboard or mobile notifications

## Requirements

- Python 3.10+
- pandas
- scikit-learn
- seaborn
- matplotlib
- statsmodels
