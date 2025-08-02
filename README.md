Electric Vehicle (EV) Adoption Forecasting using Machine Learning

📌 Overview 
As electric vehicles (EVs) become more prominent, cities and governments need tools to anticipate future demand, especially for infrastructure like charging stations. This project builds a machine learning pipeline to forecast future EV registrations based on historical data from Washington State.

🎯 Objectives 
Analyze trends in EV registrations across counties and time.

Engineer time-series and lag-based features to capture growth dynamics.

Train a regression model (Random Forest) to predict future EV counts.

Generate forecasts for the next 36 months per county.

Visualize historical and projected EV adoption curves.

🗂️ Dataset Source: Kaggle: Electric Vehicle Population Size 2024

Time span: Jan 2017 – Feb 2024

Granularity: Monthly EV registration data by county

Features:

County, Date

Battery Electric Vehicles (BEVs)

Plug-In Hybrid Electric Vehicles (PHEVs)

Electric Vehicle (EV) Total

Non-Electric Vehicle Total

Percent Electric Vehicles

🔧 Features & Engineering Time-based features:

Year, Month, Quarter, Months Since Start

Lag features:

EV totals from previous months (lag1, lag2, lag3)

Rolling statistics:

3-month rolling mean, growth slope

Growth metrics:

% change (1-month, 3-month)

📈 Visualizations EV trends over time (statewide + county)

EV vs non-EV over time

Top counties by EV count

Feature correlation heatmap

Actual vs Predicted EVs

Residual analysis

3-year forecast visualization

📊 Evaluation Metrics MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score (Explained variance)

🔮 Forecasting For a selected county (e.g., King), predict EV adoption for the next 36 months.

Rolling predictions updated using model output.

Visual comparison of historical vs forecasted EV totals.

👤 Author Vedant Salve
