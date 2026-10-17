# OmniRetail-AI Baseline Model Report

## Objective
Establish a benchmark for predicting demand_index.

## Data split
- Valid rows: 1200
- Training rows: 960
- Test rows: 240
- Features: 11
- First test date: 2026-06-30
- Last test date: 2026-08-31

## Test-set metrics
### Dummy Mean
- mae: 0.16621
- rmse: 0.19539
- mape_percent: 31.84012

### Random Forest
- mae: 0.16735
- rmse: 0.19856
- mape_percent: 31.86444

## Interpretation
Lower MAE and RMSE indicate smaller prediction errors. Compare the
Random Forest with the Dummy Regressor on the same held-out dates.

## Limitations
This dataset predicts a service demand index, not SKU-level retail
sales. Results do not establish inventory optimization or pricing impact.
MAPE may be misleading for targets close to zero.
