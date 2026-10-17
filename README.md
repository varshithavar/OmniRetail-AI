\# OmniRetail-AI



\## AI-Powered Retail Intelligence Platform



OmniRetail-AI is a retail analytics and AI project designed to support demand forecasting, dynamic pricing recommendations, and inventory replenishment alerts through a unified platform.



\## Project Objectives



\- Forecast product demand using baseline forecasting, Prophet, and LSTM models.

\- Develop pricing recommendations with configurable price constraints.

\- Identify potential stock replenishment needs.

\- Expose functionality through a FastAPI backend.

\- Track forecasting experiments and evaluation metrics with MLflow.

\- Present operational insights through a manager dashboard.



\## Proposed Architecture



1\. \*\*Data ingestion:\*\* Load the supplied CSV and JSON datasets.

2\. \*\*Data preprocessing:\*\* Validate schemas, handle missing values, and prepare features.

3\. \*\*Demand forecasting:\*\* Compare baseline, Prophet, and LSTM approaches.

4\. \*\*Dynamic pricing:\*\* Generate recommendations subject to configurable constraints.

5\. \*\*Inventory replenishment:\*\* Evaluate stock thresholds and generate alerts.

6\. \*\*API and experiment tracking:\*\* Integrate FastAPI and MLflow.

7\. \*\*Manager dashboard:\*\* Display forecasts, pricing recommendations, inventory alerts, and KPIs.



\## Repository Structure



```text

OmniRetail-AI/

├── api/

├── data/

│   ├── raw/

│   └── processed/

├── forecasting/

├── inventory/

├── notebooks/

├── pricing/

├── src/

│   └── data/

├── tests/

├── utils/

├── config.yaml

├── requirements.txt

└── README.md

```



\## Evaluation Metrics



\### Demand Forecasting

\- Mean Absolute Percentage Error (MAPE)

\- Mean Absolute Error (MAE)

\- Root Mean Squared Error (RMSE)



Initial target: MAPE below 15% on a suitable chronological holdout dataset.



\### Dynamic Pricing

\- Simulated revenue and profit changes

\- Price constraint violations

\- Evaluation on at least 20 SKUs



\### Inventory Replenishment

\- Stockout rate

\- Service level

\- Alert precision and recall, where ground-truth labels are available



\### Experiment Tracking

\- At least 10 logged MLflow experiments with parameters and evaluation metrics.



\## Research References



1\. Taylor, S. J., and Letham, B. (2018). \*Forecasting at Scale\*. https://doi.org/10.1080/00031305.2017.1380080

2\. Salinas, D. et al. \*DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks\*. https://arxiv.org/abs/1704.04110

3\. MLflow open-source project: https://github.com/mlflow/mlflow



\## Setup



Create and activate a virtual environment, then install the project dependencies:



```bash

python -m venv .venv

```



Windows CMD:



```cmd

.venv\\Scripts\\activate

pip install -r requirements.txt

```



Run the API using:



```cmd

uvicorn api.main:app --reload

```



Check the existing API routes before relying on any particular endpoint.



\## Project Status



This repository contains initial forecasting, pricing, inventory, API, data-processing, notebook, and testing components. The architecture and metrics above describe the project plan; each component and acceptance target must be validated through implementation and testing.



\## Data and Reproducibility



Keep data sources documented, avoid committing credentials or secrets, and record experiment parameters and evaluation results. Confirm that the supplied data contains the historical sales, price, and inventory fields needed for the respective modules before model training.



