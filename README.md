# Beyond Square Footage: The Impact of Macroeconomic Factors and COVID-19 on Housing Prices in Austin

**Team:** Heidi Momin · Saad Salim · Pri Baleki · Sophia Achar
**Course:** Advanced Machine Learning — Final Project (Spring 2026)

## Summary

A machine-learning pipeline that predicts Austin home sale prices (2018–2021) using property features, macroeconomic indicators (mortgage rate, CPI, unemployment), and school-quality data. We benchmark four regression models and separately analyze how the market's internal logic shifted after March 2020.

**Best model:** XGBoost — Test RMSE ≈ $68K, R² = 0.81
**Key finding:** Post-COVID prices rose 12.7% ($492K → $555K), and importance of raw space features (bathrooms, bedrooms, living area) grew sharply — consistent with remote-work demand.

## Repository contents

| File | Purpose |
|------|---------|
| `Beyond_Square_Footage_...ipynb` | Main notebook — Task A (cleaning), Task B (modeling), Task C (COVID analysis) |
| `requirements.txt` | Python dependencies |
| `data/` | Place raw CSVs here (see Data setup below) |
| `outputs/` | Generated artifacts: feature-importance CSVs, model metrics, plots |

## Data setup

This repo does not commit the raw datasets. Download them from the original sources and place in a `data/` folder:

| Dataset | Source |
|---------|--------|
| Austin housing prices | https://www.kaggle.com/datasets/ericpierce/austinhousingprices |
| 30-Year Mortgage Rate | https://fred.stlouisfed.org/series/MORTGAGE30US |
| Consumer Price Index | https://fred.stlouisfed.org/series/CPIAUCSL |
| Austin Unemployment Rate | https://fred.stlouisfed.org/series/AUST448URN |
| City of Austin Schools | https://catalog.data.gov/dataset/city-of-austin-schools-with-data |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Beyond_Square_Footage_The_Impact_of_Macroeconomic_Factors_and_COVID_19_on_Housing_Prices_in_Austin_.ipynb
```

Or open directly in Google Colab: https://colab.research.google.com/drive/13NXc4Fe9PS3WcClvSM9pIwoR28K7Kbh9

## Methodology

1. **Data cleaning** — outlier removal (1–99 percentile), median imputation, log-transform target
2. **Feature engineering** — 12+ derived features including price ratios and COVID × macro interaction terms
3. **Time-based 80/20 train-test split** — sorted by sale date to prevent lookahead bias
4. **Model benchmarking** — Linear Regression, Ridge, Random Forest, XGBoost
5. **Structural-break analysis** — separate XGBoost models for pre-COVID and post-COVID subsets
6. **Time-lag cross-correlation** — macro variables vs housing prices at lags 0–12 months

## Results

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| Linear Regression | $98,000 | 0.62 |
| Ridge Regression | $97,000 | 0.63 |
| Random Forest | $72,000 | 0.78 |
| **XGBoost** | **$68,000** | **0.81** |

## Blog

Full write-up: _[add blog URL once published]_

## References

See `Beyond_Square_Footage_Blog.docx` for full reference list.
