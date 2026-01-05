# Time Series Forecasting Model for Ethiopian Commodity Prices

## 🌾 Overview

This project presents a comprehensive machine learning approach to forecasting commodity prices in Ethiopia, specifically focusing on red onion prices. The work is conducted in collaboration with the **Agricultural Transformation Institute (ATI)** and the **Ethiopian Statistical Agency**, addressing critical challenges in agricultural price prediction for policy-making and market planning.

## 🎯 Problem Statement

Commodity price forecasting is crucial for:
- **Policy makers**: Planning agricultural policies and interventions
- **Farmers**: Making informed planting and harvesting decisions
- **Traders**: Managing inventory and pricing strategies
- **Consumers**: Understanding price trends and planning purchases

Traditional forecasting methods (ARIMA, Exponential Smoothing) struggle with:
- **Non-linear relationships**: Complex interactions between multiple factors
- **External factors**: Holidays, seasons, market shocks that don't follow simple patterns
- **Feature engineering**: Limited ability to incorporate domain knowledge

## 🔬 Research Context

While most commodity price forecasting research is conducted in regions like the **Middle East** and **China** (where extensive agricultural data infrastructure exists), this project brings advanced machine learning techniques to the Ethiopian context, adapting methodologies to local market dynamics, Ethiopian calendar systems, and regional holidays.

**Key Research Papers & References:**
- Chinese agricultural forecasting studies (often use ensemble methods with external features)
- Middle Eastern commodity price prediction (focus on oil, grains, dates)
- Time series forecasting with machine learning (LightGBM, XGBoost applications)

## 🏗️ Methodology

### Why Supervised Learning Over Traditional Forecasting Models?

#### Limitations of ARIMA and Classical Methods:
1. **Linear Assumptions**: ARIMA models assume linear relationships, missing complex non-linear patterns
2. **Limited Feature Integration**: Difficult to incorporate external features (holidays, weather, market events)
3. **Stationarity Requirements**: Require data transformation that may lose important information
4. **Manual Parameter Tuning**: ARIMA requires manual identification of p, d, q parameters
5. **Single Series Focus**: Limited ability to leverage multiple related time series

#### Advantages of Supervised Learning Approach:
1. **Non-linear Modeling**: Machine learning models (LightGBM) can capture complex, non-linear relationships
2. **Rich Feature Engineering**: Can incorporate calendar features, holidays, rolling statistics, volatility measures
3. **Automatic Feature Selection**: Models learn which features are most important
4. **Handles Missing Data**: More robust to missing values and irregular patterns
5. **Scalability**: Can easily add new features without model restructuring

### Feature Engineering Strategy

#### 1. Calendar Features Integration
Understanding the **time aspect** is crucial for commodity prices:

- **Cyclical Encoding**: Month, week, day-of-week converted to sine/cosine to capture cyclical patterns
- **Ethiopian Calendar Integration**: Ethiopian holidays (Timkat, Meskel, Ethiopian Christmas, New Year) significantly impact prices
- **Holiday Distance Features**: Distance to nearest holiday (with cyclical encoding) captures pre/post-holiday effects
- **Seasonal Indicators**: Quarterly and semiannual harvest cycles encoded as cyclical features

**Why This Matters**: Commodity prices in Ethiopia are heavily influenced by:
- Religious holidays (increased demand)
- Harvest seasons (supply fluctuations)
- Market days (weekly patterns)
- Seasonal consumption patterns

#### 2. History-Based Features
- **Rolling Statistics**: Mean, variance, volatility over multiple windows (7, 14, 30, 90, 180 days)
- **Momentum Features**: Rate of change over different periods
- **Trend Slopes**: Linear regression slopes over rolling windows
- **GARCH-like Volatility**: Captures volatility clustering (high volatility tends to persist)
- **Shock Detection**: Identifies unusual price movements
- **Price Range Features**: Position within recent price range (normalized)

#### 3. Target Encoding
- **Day-of-Week Means**: Average price by day of week
- **Month Means**: Average price by month
- Captures systematic patterns without overfitting

### Recursive vs Direct Forecasting

#### Why Recursive Forecasting?

**Recursive Forecasting** (used in this project):
- Trains **one model** that predicts the next time step
- Uses its own predictions as inputs for future steps
- **Computational Efficiency**: Single model training, fast inference
- **Scalability**: Can forecast any horizon without retraining

**Direct Forecasting**:
- Trains **separate models** for each forecast horizon (1-step, 2-step, ..., h-step)
- Each model is optimized for its specific horizon
- **Computational Cost**: Requires training h models (expensive for long horizons)
- **Memory**: Need to store h models

**Trade-off**: While direct forecasting can be more accurate (each model optimized for its horizon), recursive forecasting is more practical for:
- Real-time applications
- Long forecast horizons (30+ days)
- Resource-constrained environments
- Rapid iteration and experimentation

#### Limitations of Recursive Forecasting:

1. **Error Propagation**: Prediction errors accumulate as we forecast further into the future
   - Early prediction errors become inputs for later predictions
   - Can lead to divergence from actual values over long horizons
   
2. **Distribution Shift**: The model is trained on actual values but must predict using its own (potentially different) predictions
   - Model may not generalize well to its own output distribution
   
3. **Horizon-Dependent Accuracy**: Accuracy typically decreases as forecast horizon increases

**Mitigation Strategies Used**:
- Feature engineering that captures long-term trends
- Regime detection to adapt to different market conditions
- Robust validation on multiple horizons
- Monitoring forecast quality over different horizons

### Why LightGBM?

**LightGBM** (Light Gradient Boosting Machine) was chosen for several reasons:

1. **Performance**: 
   - Fast training and inference
   - Handles large feature sets efficiently
   - Excellent accuracy on tabular data

2. **Feature Handling**:
   - Automatic feature importance calculation
   - Handles missing values natively
   - Works well with mixed feature types

3. **Regularization**:
   - Built-in L1/L2 regularization
   - Prevents overfitting
   - Handles high-dimensional feature spaces

4. **Interpretability**:
   - Feature importance scores
   - Can identify which features drive predictions
   - Useful for domain experts (ATI, Statistical Agency)

5. **Robustness**:
   - Less sensitive to outliers than linear models
   - Handles non-linear relationships well
   - Works with small to medium datasets

**Comparison with Alternatives**:
- **XGBoost**: Similar performance but slower training
- **Random Forest**: Less accurate, more interpretable
- **Neural Networks**: Require more data, less interpretable, harder to tune
- **Linear Models**: Too simple for complex non-linear patterns

## 📊 Model Architecture

```
Input Features (150+ features)
    ↓
Feature Engineering Pipeline
    ├── Calendar Features (cyclical encoding)
    ├── Holiday Features (Ethiopian holidays)
    ├── Rolling Statistics (7, 14, 30, 90, 180 days)
    ├── Volatility & Momentum
    ├── Trend Slopes
    └── Target Encoding
    ↓
LightGBM Regressor
    ├── Lags: 1-120 days
    ├── Differentiation: 1
    └── Recursive Forecasting
    ↓
Forecast Output (30-day horizon)
```

## 📁 Project Structure

```
.
├── README.md                          # This file
├── recusrisve_forcasting_lightgbm_red_onion.ipynb  # Main notebook
├── requirements.txt                   # Python dependencies
└── data/                              # Data directory (not included in repo)
    └── df_clean_red_onion_a.csv      # Cleaned price data
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd commodity_price_forcasting

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Usage

1. **Data Preparation**: 
   - Place your cleaned data in the `data/` directory
   - Update the file path in the `load_and_prepare_data()` function

2. **Run the Notebook**:
   - Open `recusrisve_forcasting_lightgbm_red_onion.ipynb`
   - Execute cells sequentially
   - The notebook includes:
     - Data loading and preprocessing
     - Feature engineering
     - Model training
     - Evaluation and visualization
     - Future forecasting

3. **Customization**:
   - Adjust forecast horizon: `FORECAST_HORIZON = 30`
   - Modify train/validation/test splits
   - Tune LightGBM hyperparameters
   - Add new features in the feature engineering section

## 📈 Key Features

### 1. Standardized Data Loading
- `load_and_prepare_data()`: Handles data loading, date conversion, train/test splitting

### 2. Comprehensive Feature Engineering
- `build_all_features()`: Creates all features for training
- `build_future_features()`: Creates features for inference (ensures consistency)

### 3. Evaluation Metrics
- `compute_metrics()`: Calculates MAE, MAPE, RMSE, R²
- `plot_forecast_comparison()`: Visualizes actual vs predicted
- `plot_residuals_analysis()`: Diagnostic plots for model evaluation

### 4. Visualization
- `plot_feature_importance()`: Top N feature importances
- `plot_out_of_sample_forecast()`: Comprehensive forecast visualization with residuals

## 🔍 Model Performance

The model achieves strong performance on out-of-sample data:
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination

(Actual metrics will be displayed when you run the notebook)

## 🤝 Collaboration

This project is developed in collaboration with:
- **Agricultural Transformation Institute (ATI)**: Policy and agricultural development
- **Ethiopian Statistical Agency**: Data and statistical methodology

## 📚 References & Further Reading

### Research Papers
- Commodity price forecasting in developing economies
- Machine learning for time series forecasting
- Feature engineering for agricultural data
- Recursive vs direct forecasting methodologies

### Related Work
- Chinese agricultural forecasting systems
- Middle Eastern commodity price prediction models
- Time series forecasting with gradient boosting

## ⚠️ Limitations & Future Work

### Current Limitations:
1. **Error Propagation**: Recursive forecasting accumulates errors over long horizons
2. **Data Requirements**: Requires sufficient historical data for feature engineering
3. **External Factors**: Limited integration of weather, supply chain disruptions
4. **Market Regimes**: Could benefit from more sophisticated regime detection

### Future Improvements:
1. **Hybrid Approaches**: Combine recursive and direct forecasting
2. **External Data**: Integrate weather, trade, and policy data
3. **Ensemble Methods**: Combine multiple models
4. **Online Learning**: Update model as new data arrives
5. **Uncertainty Quantification**: Provide prediction intervals, not just point forecasts

## 📝 License

[Specify your license here]

## 👥 Authors

- Developed for ATI and Ethiopian Statistical Agency
- Contact: musshaile@gmail.com

## 🙏 Acknowledgments

- Agricultural Transformation Institute (ATI)
- Ethiopian Statistical Agency
- Research community working on commodity price forecasting

---

**Note**: This notebook is designed to work without the actual data file. Users should provide their own cleaned time series data following the expected format (date column and price column).

