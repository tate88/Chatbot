from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import requests
import pyodbc
import re
import os
from dotenv import load_dotenv
import secrets
from fuzzywuzzy import fuzz
from tabulate import tabulate
import locale
import decimal
from collections import defaultdict
import pandas as pd
from datetime import datetime
import traceback
import logging
import matplotlib  # Must be imported first
matplotlib.use('Agg')  # Set backend *before* importing pyplot or Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import random
from prophet import Prophet
import matplotlib.pyplot as plt  # Safe to import after backend is set
import base64
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# 🔒 Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)  # Optional: for Python's built-in random

locale.setlocale(locale.LC_ALL, '')

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
socketio = SocketIO(app)

user_histories = defaultdict(list)  # session_id -> list of (role, message)

FORECAST_KEYWORDS = [
    'forecast', 'predict', 'projection', 'outlook',
    'next month', 'next quarter', 'coming year',
    'future trend', 'estimate future', 'upcoming months', 'expected sales', 'expected revenue',
    'anticipate', 'future sales', 'sales trend', 'projected growth', 'market prediction',
    'sales forecast', 'revenue projection', 'future performance', 'growth estimate',
    'sales outlook', 'next year', 'financial forecast', 'business projection',
    'demand forecast', 'revenue estimate', 'sales expectation', 'future demand',
    'projected revenue', 'projected earnings', 'trend analysis'
]

def is_forecast_question(user_input):
    """Determine if user query is a forecast question using LLM."""
    try:
        prompt = f"""
        Determine if this query is asking for a sales forecast or prediction of future values.
        Examples of forecast questions:
        - "What will sales be next month?"
        - "Predict revenue for Q4"
        - "Forecast performance for the coming year"
        - "How will product X sell in 2024?"
        - "What's the sales outlook for next quarter?"

        Answer with only 'yes' or 'no'.
        
        Query: "{user_input}"
        """

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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"
            },
            json={
                "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                "messages": [
                    {"role": "system", "content": "You determine if questions are about forecasting."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 10,
            },
        )
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip().lower()
            return 'yes' in answer
        
        # Fallback to keyword matching if API fails to return expected format
        return any(keyword in user_input.lower() for keyword in FORECAST_KEYWORDS)
        
    except Exception as e:
        print(f"Error in forecast detection: {str(e)}")
        # Fallback to the original keyword matching approach
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in FORECAST_KEYWORDS) or \
               any(fuzz.ratio(user_input_lower, keyword) > 75 for keyword in FORECAST_KEYWORDS)


def fetch_forecast_data():
    conn = pyodbc.connect(
        r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=EBuilder;Trusted_Connection=yes;"
    )

    query = """
    SELECT 
        DATEFROMPARTS(YEAR(TrxDate), MONTH(TrxDate), 1) AS ds,
        SUM(DISTINCT TrxAmt) AS y
    FROM 
        var_trx_sales_analysis_by_ar
    WHERE 
        TrxDate IS NOT NULL 
        AND TrxAmt IS NOT NULL
    GROUP BY 
        YEAR(TrxDate), MONTH(TrxDate)
    ORDER BY 
        ds
"""


    df = pd.read_sql(query, conn)
    df['ds'] = pd.to_datetime(df['ds'])
    print(df)
    return df

def forecast_sales(df, periods):
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    return model,result


def hybrid_forecast(df, periods):
    # Input validation
    if df.empty or 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("DataFrame must have 'ds' and 'y' columns")
    
    print(f"🔍 Starting hybrid forecast for {periods} periods...")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # 1. Log-transform the target
    print(f"Original data range: {df['y'].min():.2f} to {df['y'].max():.2f}")
    
    df.loc[df['y'] <= 0, 'y'] = None  # 设为缺失值

    
    if df['y'].max() < 100:
        print("⚠️ Detected already log-transformed values. Skipping log1p.")
        df['y_original'] = np.expm1(df['y'])
    else:
        print("✅ Applying log1p transformation...")
        df['y_original'] = df['y'].copy()
        df['y'] = df['y'].clip(lower=1)
        df['y'] = np.log1p(df['y'])
    
    print(f"Transformed data range: {df['y'].min():.2f} to {df['y'].max():.2f}")
    
    # 2. Run Prophet with your improved parameters
    prophet_model = Prophet(
        growth='linear',
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_range=0.8,
        n_changepoints=25,
    )
    
    prophet_model.fit(df)
    
    # 3. Generate forecast
    future = prophet_model.make_future_dataframe(periods, freq='MS')
    forecast = prophet_model.predict(future)
    
    print(f"Prophet forecast range (log scale): {forecast['yhat'].min():.2f} to {forecast['yhat'].max():.2f}")
    
    # 4. Apply LSTM to residuals (if you have this function)
    try:
        adjusted_forecast = apply_lstm_on_residuals(df, forecast)
        print("✅ LSTM applied to residuals")
    except (NameError, AttributeError):
        print("⚠️ apply_lstm_on_residuals not found - using Prophet only")
        adjusted_forecast = forecast.copy()
        adjusted_forecast['adjusted_yhat'] = forecast['yhat']
    
    # 5. **CRITICAL**: Reverse transformations to original scale
    print("🔄 Converting back to original scale...")
    
    transform_columns = ['yhat', 'yhat_lower', 'yhat_upper', 'adjusted_yhat']
    
    for col in transform_columns:
        if col in adjusted_forecast.columns:
            # Convert from log scale back to original scale
            adjusted_forecast[col] = np.expm1(adjusted_forecast[col])
            # Clip to prevent negative values
            adjusted_forecast[col] = adjusted_forecast[col].clip(lower=0)
            print(f"✅ {col}: {adjusted_forecast[col].min():.0f} to {adjusted_forecast[col].max():.0f}")
    
    # 6. Summary insights
    if len(df) > 0:
        last_actual = df['y_original'].iloc[-1]
        first_forecast_idx = len(df)
        if first_forecast_idx < len(adjusted_forecast):
            first_forecast = adjusted_forecast['adjusted_yhat'].iloc[first_forecast_idx]
            change_pct = ((first_forecast - last_actual) / last_actual) * 100
            print(f"📊 First forecast vs last actual: {first_forecast:.0f} vs {last_actual:.0f} ({change_pct:.1f}% change)")
    
    return {
        'prophet_model': prophet_model,
        'adjusted_forecast': adjusted_forecast,
        'historical_data': df
    }

def apply_lstm_on_residuals(df, forecast_df):
    """
    Enhanced LSTM application to Prophet residuals with multiple improvements:
    - Better feature engineering
    - Improved model architecture
    - Cross-validation
    - Ensemble approach
    - Better evaluation metrics
    """
    try:
        print("\n🚀 Starting Enhanced LSTM v3 processing...")
        
        # Minimum data check
        if len(df) < 30:
            print("⚠️ Not enough data for LSTM. Returning original forecast.")
            forecast_df['adjusted_yhat'] = forecast_df['yhat']
            return forecast_df
        
        if df[['ds', 'y']].isnull().any().any():
            print("⚠️ Found NaNs in 'ds' or 'y' columns. Dropping...")
            df = df.dropna(subset=['ds', 'y'])

        if 'residual' in df.columns and df['residual'].isnull().any():
            print("⚠️ Found NaNs in residuals. Dropping...")
            df = df.dropna(subset=['residual'])
            
        
        
        # Enhanced feature engineering
        df_enhanced = create_enhanced_features(df.copy())
        forecast_enhanced = create_enhanced_features(forecast_df.copy())
        
        # Use walk-forward validation for better accuracy assessment
        results = walk_forward_validation(df_enhanced, n_splits=3)
        prophet_accuracy = results['prophet_accuracy']
        print(f"Prophet Accuracy: {results['prophet_accuracy']:.2f}%")
      
        if not results['use_lstm']:
            print("⚠️ LSTM doesn't improve accuracy. Using Prophet only.")
            forecast_df['adjusted_yhat'] = forecast_df['yhat']
            return forecast_df
        
        # Train final model on all data
        final_forecast = train_final_model(df_enhanced, forecast_enhanced, results['best_params'])
        
        # Apply ensemble approach
        ensemble_forecast = apply_ensemble(forecast_df, final_forecast)
   
        
        print("✅ Enhanced LSTM v3 completed successfully.")
        return ensemble_forecast
        
    except Exception as e:
        print(f"❌ Error in LSTM v3: {str(e)}")
        forecast_df['adjusted_yhat'] = forecast_df['yhat']
        return forecast_df

def create_enhanced_features(df):
    """Create comprehensive feature set for better LSTM performance"""
    df = df.copy()
    
    # Temporal features
    df['month'] = df['ds'].dt.month
    df['day_of_year'] = df['ds'].dt.dayofyear / 365.25
    df['quarter'] = df['ds'].dt.quarter
    df['week_of_year'] = df['ds'].dt.isocalendar().week / 52.0
    df['day_of_week'] = df['ds'].dt.dayofweek
    
    # Cyclical encoding for better seasonal representation
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lag features (if target exists)
    if 'y' in df.columns:
        for lag in [1, 7, 30]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            if len(df) > window:
                df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()
                
    if 'is_zero_filled' not in df.columns:
        df['is_zero_filled'] = 0
    
    return df

def walk_forward_validation(df, n_splits=3):
    """
    Perform walk-forward validation to assess LSTM effectiveness.
    Calculates and returns forecast accuracy percentages for Prophet and LSTM.
    """

    print("🔄 Performing walk-forward validation...")

    prophet_errors = []
    lstm_errors = []

    total_len = len(df)
    test_size = max(30, total_len // (n_splits + 1))

    best_params = {
        'sequence_length': 12,
        'lstm_units': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    }

    for i in range(n_splits):
        split_point = total_len - (n_splits - i) * test_size
        train_data = df.iloc[:split_point]
        test_data = df.iloc[split_point:split_point + test_size]

        if len(train_data) < 30 or len(test_data) < 5:
            continue

        # Prophet forecast
        prophet_model = Prophet(
            growth='linear',
            seasonality_mode='additive',
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=10.0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_range=0.8,
            n_changepoints=25,
        )
        prophet_model.fit(train_data[['ds', 'y']])
        prophet_pred = prophet_model.predict(test_data[['ds']])
        prophet_error = mean_absolute_error(test_data['y'], prophet_pred['yhat'])
        prophet_errors.append(prophet_error)

        # LSTM forecast (residual adjustment)
        try:
            lstm_pred = train_lstm_model(train_data, test_data, prophet_pred, best_params)
            lstm_error = mean_absolute_error(test_data['y'], lstm_pred)
            lstm_errors.append(lstm_error)
        except:
            lstm_errors.append(prophet_error * 1.1)  # Penalize LSTM failure

    # Compute averages
    avg_prophet_error = np.mean(prophet_errors)
    avg_lstm_error = np.mean(lstm_errors)
    avg_y = df['y'].mean()

    # Calculate approximate accuracy %
    prophet_accuracy = 100 - (avg_prophet_error / avg_y * 100)
    lstm_accuracy = 100 - (avg_lstm_error / avg_y * 100)
    improvement = lstm_accuracy - prophet_accuracy

    print("\n✅ Final Forecast Accuracy:")
    print(f"📈 Prophet Accuracy: {prophet_accuracy:.2f}%")
    print(f"📈 LSTM Accuracy:    {lstm_accuracy:.2f}%")
    print(f"📊 Accuracy Improvement: {improvement:.2f}%\n")

    return {
        'use_lstm': improvement > 0.01,
        'prophet_accuracy': prophet_accuracy,
        'lstm_accuracy': lstm_accuracy,
        'improvement': improvement,
        'best_params': best_params
    }

def train_lstm_model(train_data, test_data, prophet_pred, params):
    """Train LSTM model on residuals with enhanced architecture"""
    
    # Calculate residuals
    train_prophet = Prophet(
        growth='linear',
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_range=0.8,
        n_changepoints=25,
    )
    train_prophet.fit(train_data[['ds', 'y']])
    train_pred = train_prophet.predict(train_data[['ds']])
    
    residuals = train_data['y'].values - train_pred['yhat'].values
    
    # Prepare features
    feature_cols = [
    'month_sin', 'month_cos', 'day_sin', 'day_cos', 'day_of_year', 'is_zero_filled'
]

    
    # Scale residuals
    scaler = StandardScaler()  # Use StandardScaler for better performance
    residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequence_length = params['sequence_length']
    X, y = [], []
    
    for i in range(sequence_length, len(residuals_scaled)):
        # Residual sequence
        seq = residuals_scaled[i-sequence_length:i]
        
        # Additional features
        features = train_data[feature_cols].iloc[i].values
        
        # Combine sequence with features
        X.append(np.concatenate([seq, features]))
        y.append(residuals_scaled[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM
    feature_dim = len(feature_cols)
    X = X.reshape(X.shape[0], sequence_length + feature_dim, 1)
    
    # Build enhanced LSTM model
    model = Sequential([
        Bidirectional(LSTM(params['lstm_units'], 
                          return_sequences=True, 
                          dropout=params['dropout_rate']),
                     input_shape=(sequence_length + feature_dim, 1)),
        BatchNormalization(),
        Bidirectional(LSTM(params['lstm_units'] // 2, 
                          dropout=params['dropout_rate'])),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(params['dropout_rate']),
        Dense(1)
    ])
    
    # Compile with custom optimizer
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train model
    model.fit(X, y, 
              epochs=100, 
              batch_size=min(32, len(X) // 4),
              verbose=0,
              callbacks=callbacks)
    
    # Predict on test data
    test_residuals = test_data['y'].values - prophet_pred['yhat'].values
    test_residuals_scaled = scaler.transform(test_residuals.reshape(-1, 1)).flatten()
    
    # Create test sequences
    X_test = []
    last_sequence = residuals_scaled[-sequence_length:]
    
    for i in range(len(test_data)):
        features = test_data[feature_cols].iloc[i].values
        X_test.append(np.concatenate([last_sequence, features]))
        
        # Update sequence for next prediction
        if i < len(test_residuals_scaled):
            last_sequence = np.append(last_sequence[1:], test_residuals_scaled[i])
    
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], sequence_length + feature_dim, 1)
    
    # Predict and inverse transform
    residual_pred = model.predict(X_test, verbose=0)
    residual_pred = scaler.inverse_transform(residual_pred).flatten()
    
    # Combine with Prophet predictions
    final_pred = prophet_pred['yhat'].values + residual_pred
    
    return final_pred

def train_final_model(df, forecast_df, params):
    """Train the final LSTM model on all available data"""
    print("🎯 Training final model on all data...")
    
    # Train Prophet on full data
    prophet_model = Prophet(
        growth='linear',
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_range=0.8,
        n_changepoints=25,
    )
    prophet_model.fit(df[['ds', 'y']])
    
    # Get Prophet predictions for historical data
    prophet_pred = prophet_model.predict(df[['ds']])
    residuals = df['y'].values - prophet_pred['yhat'].values
    
    # Train LSTM on residuals
    feature_cols = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'day_of_year']
    
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequence_length = params['sequence_length']
    X, y = [], []
    
    for i in range(sequence_length, len(residuals_scaled)):
        seq = residuals_scaled[i-sequence_length:i]
        features = df[feature_cols].iloc[i].values
        X.append(np.concatenate([seq, features]))
        y.append(residuals_scaled[i])
    
    X = np.array(X)
    y = np.array(y)
    
    feature_dim = len(feature_cols)
    X = X.reshape(X.shape[0], sequence_length + feature_dim, 1)
    
    # Build and train model
    model = Sequential([
        Bidirectional(LSTM(params['lstm_units'], 
                          return_sequences=True, 
                          dropout=params['dropout_rate']),
                     input_shape=(sequence_length + feature_dim, 1)),
        BatchNormalization(),
        Bidirectional(LSTM(params['lstm_units'] // 2, 
                          dropout=params['dropout_rate'])),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(params['dropout_rate']),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    callbacks = [
        EarlyStopping(monitor='loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6)
    ]
    
    model.fit(X, y, 
              epochs=150, 
              batch_size=min(32, len(X) // 4),
              verbose=0,
              callbacks=callbacks)
    
    # Predict future residuals
    future_steps = len(forecast_df) - len(df)
    last_sequence = residuals_scaled[-sequence_length:]
    future_residuals = []
    
    for i in range(future_steps):
        future_idx = len(df) + i
        features = forecast_df[feature_cols].iloc[future_idx].values
        X_future = np.concatenate([last_sequence, features])
        X_future = X_future.reshape(1, sequence_length + feature_dim, 1)
        
        pred = model.predict(X_future, verbose=0)[0, 0]
        future_residuals.append(pred)
        
        # Update sequence
        last_sequence = np.append(last_sequence[1:], pred)
    
    # Inverse transform
    future_residuals = scaler.inverse_transform(
        np.array(future_residuals).reshape(-1, 1)
    ).flatten()
    
    # Apply to forecast
    forecast_df['lstm_adjustment'] = 0
    forecast_df.loc[len(df):, 'lstm_adjustment'] = future_residuals

    

    
    return forecast_df

def apply_ensemble(original_forecast, lstm_forecast):
    """Apply ensemble approach for robust predictions"""
    original_forecast['adjusted_yhat'] = original_forecast['yhat'].copy()
    
    # Apply LSTM adjustment with dampening for far future
    future_mask = original_forecast.index >= len(lstm_forecast) - (len(original_forecast) - len(lstm_forecast))
    
    if 'lstm_adjustment' in lstm_forecast.columns:
        # Apply dampening factor for far future predictions
        adjustment = lstm_forecast['lstm_adjustment'].values
        
        # Dampen adjustments for far future (reduce impact over time)
        future_steps = len(adjustment[adjustment != 0])
        if future_steps > 0:
            damping_factor = np.exp(-np.arange(future_steps) / (future_steps * 0.3))
            adjustment[adjustment != 0] *= damping_factor
        
        original_forecast['adjusted_yhat'] += adjustment
    
    return original_forecast

# Additional utility functions for hyperparameter tuning
def optimize_hyperparameters(df, n_trials=20):
    """Optimize LSTM hyperparameters using random search"""
    print("🔧 Optimizing hyperparameters...")
    
    param_grid = {
        'sequence_length': [6, 12, 18, 24],
        'lstm_units': [32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.002]
    }
    
    best_score = float('inf')
    best_params = None
    
    for _ in range(n_trials):
        params = {
            key: np.random.choice(values) 
            for key, values in param_grid.items()
        }
        
        try:
            results = walk_forward_validation(df, n_splits=2)
            if results['lstm_error'] < best_score:
                best_score = results['lstm_error']
                best_params = params
        except:
            continue
    
    return best_params or {
        'sequence_length': 12,
        'lstm_units': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    }
    



def is_product_forecast_question(user_input):
    """Enhanced detection for product-specific forecast questions"""
    user_input_lower = user_input.lower()
    
    # Check for product-specific forecast patterns
    product_patterns = [
        r'forecast.*(?:for|of)\s+(\w+)',
        r'predict.*sales.*(?:for|of)\s+(\w+)',
        r'(\w+).*sales.*forecast',
        r'what.*will.*(\w+).*sell',
        r'(\w+).*future.*sales'
    ]
    
    for pattern in product_patterns:
        if re.search(pattern, user_input_lower):
            return True
    
    return any(keyword in user_input_lower for keyword in FORECAST_KEYWORDS)


def generate_product_forecast_sql(user_input, product_hint=None):
    """Generate SQL query specifically for product sales data without summing TrxAmt"""
    schema = get_database_schema()
    schema_text = "\n".join(
        f"{table}: {', '.join(info['columns'])}" for table, info in schema.items()
    )

    relationship_hint = """
    CRITICAL CHANGES FOR THIS QUERY:
- DO NOT SUM TrxAmt directly due to possible duplicate TrxNo values
- Use SELECT DISTINCT TrxNo, TrxDate, TrxAmt when working with raw transactions
- If the user explicitly asks for monthly or aggregated data:
    → Convert TrxDate to month (e.g., FORMAT(TrxDate, 'yyyy-MM') or DATEFROMPARTS(YEAR, MONTH, 1))
    → Group by the derived month
    → Sum TrxAmt *after* using DISTINCT to avoid duplicates

For PRODUCT SALES FORECASTING, use these relationships:
- var_trx_sales_analysis_by_ar contains: TrxDate (date), TrxAmt (sales amount), TrxStkId (product ID), StkCode (Stock Code), StkCat3 (brand/category)
- Filter by product using TrxStkDesc1 LIKE '%product%' or StkCat3 = 'BRAND'
- Include columns: TrxDate (or derived month as 'ds') and TrxAmt (or sum as 'y')
- Ensure WHERE TrxDate IS NOT NULL AND TrxAmt IS NOT NULL
"""




    prompt = f"""
You are a SQL expert specialized in PRODUCT SALES FORECASTING.

💡 Goal:
Generate a SQL query that retrieves historical monthly sales for a **single product** (given its StkCode) to be used for Prophet time series forecasting.

✅ Strict Requirements:
1. Output **two columns only**:
   - `ds`: first day of the transaction month (e.g., 2023-05-01)
   - `y`: total sales amount for that month
2. Use `var_trx_sales_analysis_by_ar` as main source for:
   - TrxDate
   - TrxAmt
   - TrxNo
   - StkCode
3. Filter:
   - WHERE StkCode = '{product_hint}'
   - AND TrxDate IS NOT NULL
   - AND TrxAmt IS NOT NULL
4. De-duplicate:
   - Use `SELECT DISTINCT TrxNo, TrxDate, TrxAmt` to eliminate duplicate transactions
5. Aggregation:
   - Group by YEAR(TrxDate), MONTH(TrxDate)
   - `ds` = DATEFROMPARTS(YEAR(TrxDate), MONTH(TrxDate), 1)
   - `y` = SUM(TrxAmt)
6. ORDER BY ds ASC

User request summary: "{user_input}"

Return ONLY the SQL query. DO NOT explain or add any comments.
"""



    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": [
                    {"role": "system", "content": "Generate SQL queries for product sales forecasting. Return only SQL."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 800,
            },
        )
        data = response.json()
        
        if not data.get("choices"):
            return None
            
        sql = data["choices"][0]["message"]["content"].strip()
        
        # Clean SQL
        if sql.startswith("```sql"):
            sql = sql.replace("```sql", "").replace("```", "").strip()
        elif sql.startswith("```"):
            sql = sql.replace("```", "").strip()


        print(f"Generated SQL:\n{sql}\n")
        
        return sql.replace("`", "").strip()
        
    except Exception as e:
        print(f"Error generating product forecast SQL: {e}")
        return None
    
def reference_product_sql(user_input):

    schema = get_database_schema()
    schema_text = "\n".join(
        f"{table}: {', '.join(info['columns'])}" for table, info in schema.items()
    )

    relationship_hint = """
    IMPORTANT GUIDELINES FOR QUERY DESIGN:
- Avoid directly summing TrxAmt, as TrxNo values may be duplicated
- When working with raw transactions data, always use: SELECT DISTINCT TrxNo, TrxDate, TrxAmt.
- If the user explicitly asks for monthly or aggregated data:
    → Convert TrxDate to month (e.g., FORMAT(TrxDate, 'yyyy-MM') or DATEFROMPARTS(YEAR, MONTH, 1))
    → Group by the derived month
    → Sum TrxAmt *after* using DISTINCT to avoid duplicates

For PRODUCT SALES FORECASTING, use these relationships:
- var_trx_sales_analysis_by_ar contains: TrxDate (date), TrxAmt (sales amount), TrxStkId (product ID), StkCode (Stock Code)
- Filter by product using StkName2 = "(SST 5%) KETUPAT"
- Include columns: TrxDate (or derived month as 'ds') and TrxAmt (or sum as 'y')
- Ensure WHERE TrxDate IS NOT NULL AND TrxAmt IS NOT NULL
- The product weight can be retrieved by joining on TrxStkId = StkId in the STK_MTN table and selecting StkWeigth1
"""


    prompt = f"""
You are a SQL expert for PRODUCT SALES FORECASTING. Generate a query that returns historical sales data for Prophet forecasting.

Database Schema:
{schema_text}

{relationship_hint}

REQUIREMENTS for forecasting queries:
1. Return columns named:
   - 'ds' → First day of each month
   - 'y' → Monthly total sales amount
   - 'weight' → Product size in KG (from StkWeigth1)
2. Filter by StkName2 = '(SST 5%) KETUPAT'
3. Include only TrxDate and TrxAmt that are NOT NULL
4. Use a subquery to SELECT DISTINCT TrxNo, TrxDate, TrxAmt, TrxStkID to eliminate duplication
5. Join STK_MTN on TrxStkId = StkID
6. Use DATEFROMPARTS(YEAR(TrxDate), MONTH(TrxDate), 1) as 'ds'
7. Group by year, month, and weight
8. Order by ds and weight
9. Do not combine weights — keep 5KG and 10KG separate
10. Output ONLY the SQL query



User request: "{user_input}"

Return ONLY the SQL query:
"""


    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": [
                    {"role": "system", "content": "Generate SQL queries for product sales forecasting. Return only SQL."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 1000,
            },
        )
        data = response.json()
        
        if not data.get("choices"):
            return None
            
        sql = data["choices"][0]["message"]["content"].strip()
        
        # Clean SQL
        if sql.startswith("```sql"):
            sql = sql.replace("```sql", "").replace("```", "").strip()
        elif sql.startswith("```"):
            sql = sql.replace("```", "").strip()


        print(f"Generated SQL:\n{sql}\n")
        
        return sql.replace("`", "").strip()
        
    except Exception as e:
        print(f"Error generating product forecast SQL: {e}")
        return None

def fetch_product_forecast_data(sql_query):
    """Fetch product-specific sales data and aggregate it monthly"""
    try:
        conn = pyodbc.connect(
            r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=EBuilder;Trusted_Connection=yes;"
        )
        df = pd.read_sql(sql_query, conn)

        if df.empty:
            print("⚠️ Query returned no data.")
            return None

        if 'ds' in df.columns and 'y' in df.columns:
            df['ds'] = pd.to_datetime(df['ds']).dt.normalize()
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['y'] = df['y'].apply(lambda x: max(0, x))
            df = df.dropna()

        if not df.empty:
                # Convert to month start
                df['ds'] = df['ds'].values.astype('datetime64[M]')
                df = df.groupby('ds', as_index=False)['y'].sum()

              
                full_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='MS')
                df = df.set_index('ds').reindex(full_range, fill_value=0).reset_index()
                df.columns = ['ds', 'y']  # Rename back



        print("\n📄 Monthly Aggregated Data Preview:")
        print(df)

        return df if not df.empty else None

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

    finally:
        try:
            conn.close()
        except:
            pass




def generate_product_forecast_summary(forecast_df, days):
    """Generate a summary specifically for product forecasting."""
    if forecast_df.empty:
        return f"No forecast data available ."

    recent_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(min(days, len(forecast_df)))

    avg_forecast = recent_data['yhat'].mean()
    min_forecast = recent_data['yhat_lower'].min()
    max_forecast = recent_data['yhat_upper'].max()

    # Determine trend direction
    if len(recent_data) > 1:
        delta = recent_data['yhat'].iloc[-1] - recent_data['yhat'].iloc[0]
        if abs(delta) < 0.05 * recent_data['yhat'].iloc[0]:
            trend = "relatively stable"
        elif delta > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "stable"

    # Describe uncertainty
    range_width = max_forecast - min_forecast
    relative_uncertainty = range_width / max(1, abs(avg_forecast))
    if relative_uncertainty > 1.0:
        uncertainty_msg = "The forecast shows high uncertainty with wide fluctuations expected."
    elif relative_uncertainty > 0.3:
        uncertainty_msg = "The forecast has moderate uncertainty with some variation in daily sales."
    else:
        uncertainty_msg = "The forecast shows low uncertainty and relatively stable sales patterns."

    # Handle potential negative values
    if min_forecast < 0:
        note_negative = "⚠️ Some predicted values are negative, which may indicate returns, refunds, or anomalies."
    else:
        note_negative = ""

    summary = f"""


- **Trend:** Sales are expected to be {trend}.
- **Expected daily sales range:** ${min_forecast:,.0f} to ${max_forecast:,.0f}.
- **Average forecasted sales:** ${avg_forecast:,.0f} per day.
- {uncertainty_msg}
{note_negative}
"""
    return summary.strip()





def generate_forecast_summary(user_input,summary):
    prompt = f"""
You are a skilled business analyst tasked with generating actionable insights based on sales forecasts extracted from an ERP system.

Context:
- The following is the user input that triggered this analysis:
{user_input}

-The following is a summary of the sales forecast data:
{summary}

Instructions:
1. If the forecast is for an existing product:
   - Analyze key trends and patterns in the forecast, considering how this information would be used in an ERP-managed environment.
   - Identify potential business implications or inventory management challenges.
   - Recommend three to four specific and actionable strategies that leverage ERP capabilities such as inventory planning, automatic reordering, or supply chain optimization.
   - Explain the expected business benefits or return on investment from implementing these strategies.

2. If the forecast is being used as a reference for a new product:
   - Do not analyze trends or patterns from the forecast.
   - Focus only on providing marketing and launch inventory recommendations based on the reference product’s performance.
   - Highlight key risks or assumptions when using another product’s forecast as a basis for decision-making.

Formatting instructions:
- For existing products, organize the response using sections such as "Forecast Trends", "Business Impact", and "Recommended Actions".
- For reference-based new products, provide a clear explanation in paragraph form only. Do not use section headers. Do not analyze trends. Focus on marketing suggestions and inventory guidance.
- Use full sentences throughout.
- Avoid bullet points, bold text, or markdown.
- Ensure the response is clean and appropriate for plain-text or HTML display.
"""





    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": [
                    {"role": "system", "content": "You are a data analyst who writes detailed business-focused summaries."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
        )
        response.raise_for_status()
        enhanced_summary = response.json()["choices"][0]["message"]["content"].strip()
        
        # Format the response for better display in HTML
        formatted_summary = enhanced_summary.replace('\n\n', '<br><br>').replace('\n', '<br>')
        return formatted_summary
    except Exception as e:
        print(f"Error generating enhanced forecast summary: {str(e)}")
        return summary  # Return original summary if enhancement fails

    




def extract_forecast_period(user_input: str) -> int:
    """
    Use LLM to extract forecast period in months from user input.
    Falls back to regex if LLM fails.
    """
    try:
        prompt = f"""
        Extract the forecast period in months from the following user request.
        If the user asks for a forecast for a specific year, month, or period, convert it to the number of months.
        Only return a single integer (number of months). Do not explain.

        Example requests:
        - "Forecast sales for the next 6 months" → 6
        - "Predict revenue in 2026" → (2026 - current year) * 12
        - "Show outlook for next year" → 12
        - "Estimate for next 2 years" → 24

        User request: "{user_input}"
        """
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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"
            },
            json={
                "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                "messages": [
                    {"role": "system", "content": "Extract forecast period in months as an integer."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 10,
            },
        )
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip()
            # Try to extract integer from LLM response
            match = re.search(r'\d+', answer)
            if match:
                months = int(match.group(0))
                return max(1, months)
    except Exception as e:
        print(f"LLM extraction failed: {str(e)}")

    # Fallback to regex extraction
    user_input = user_input.lower()
    user_input = re.sub(r'\s+', ' ', user_input)
    user_input = re.sub(r'in the next', 'next', user_input)

    year_match = re.search(r"(?:in|for)\s+(20\d{2})", user_input)
    if year_match:
        target_year = int(year_match.group(1))
        current_year = datetime.now().year
        months = (target_year - current_year) * 12
        return max(1, months)

    match = re.search(r'(?:next|in)\s*(\d+)\s*(weeks?|months?|years?)', user_input)
    if match:
        number = int(match.group(1))
        unit = match.group(2).rstrip('s')
        conversion = {"week": 0.25, "month": 1, "year": 12}
        months = round(number * conversion[unit])
        return max(1, months)

    if "next week" in user_input:
        return 1
    elif "next month" in user_input:
        return 1
    elif "next year" in user_input:
        return 12

    match = re.search(r'\b(\d+)\s*(?:months?|years?|weeks?)?\b', user_input)
    if match:
        months = int(match.group(1))
        return max(1, months)

    return 1


def save_forecast_plot(df, forecast_df, model, filename="static/forecast_plot.png"):
    """
    Enhanced forecast visualization with:
    - Clear historical data markers
    - Improved forecast visualization
    - Better date handling
    - Professional styling
    - Skip LSTM line if not significant
    """
    plt.figure(figsize=(14, 7))
    
    # 1. Plot historical data
    plt.scatter(
        df['ds'], df['y_original'],
        color='black', s=40,
        label='Actual Sales',
        zorder=3
    )
    
    # 2. Plot Prophet's forecast components
    model.plot(forecast_df, ax=plt.gca(), uncertainty=True, plot_cap=False)
    
    # 3. Highlight forecast period
    last_history_date = df['ds'].max()
    forecast_period = forecast_df[forecast_df['ds'] > last_history_date]
    
    # 4. Plot LSTM-adjusted forecast only if available and significantly different
    if 'adjusted_yhat' in forecast_df.columns:
        # Check if LSTM made a significant difference (>3% change)
        prophet_forecast = forecast_period['yhat'].mean()
        lstm_forecast = forecast_period['adjusted_yhat'].mean()
        pct_diff = abs((lstm_forecast - prophet_forecast) / prophet_forecast) * 100
        
        if pct_diff > 3:
            plt.plot(
                forecast_period['ds'], forecast_period['adjusted_yhat'],
                'r--', linewidth=2.5,
                label='LSTM-Adjusted Forecast'
            )
            print(f"✅ LSTM adjustment shown (made {pct_diff:.1f}% difference)")
        else:
            print(f"ℹ️ LSTM adjustment skipped (only {pct_diff:.1f}% difference)")
    
    # 5. Add forecast start indicator
    plt.axvline(
        x=last_history_date,
        color='red', linestyle=':', alpha=0.7,
        label='Forecast Start'
    )
    
    # 6. Customize plot appearance
    plt.title(f"Sales Forecast\nLast Updated: {last_history_date.strftime('%Y-%m-%d')}", pad=20)
    plt.xlabel("Date", labelpad=10)
    plt.ylabel("Sales Amount", labelpad=10)
    plt.legend(loc='upper left', framealpha=1)
    plt.grid(True, alpha=0.3)
    
    # 7. Improve date formatting
    plt.gcf().autofmt_xdate()
    
    # 8. Ensure directory exists and save
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight', dpi=120)
    plt.close()
    
    print(f"✅ Enhanced forecast plot saved to {filename}")
    return filename


def generate_hybrid_forecast_summary(forecast_df, months, detailed=False, reference=False,historical_df=None):
    """
    Generates a summary comparing Prophet and LSTM-adjusted forecasts (monthly version).
    If used as a reference for a new product, skips trend analysis and outputs only marketing guidance.
    """
    if forecast_df.empty:
        return "No forecast data available."

    if reference:
        if historical_df is None or historical_df.empty:
            return "No historical data available for reference forecasting."
            
        # Extract reference product data
        historical = historical_df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        # Print out the historical data
        print("📊 Historical Data:")
        print(historical)
        print(f"\nData shape: {historical.shape}")
        print(f"Date range: {historical['ds'].min()} to {historical['ds'].max()}")
        print(f"Sales range: RM{historical['y'].min():,.0f} to RM{historical['y'].max():,.0f}")
    
        # Optional: Print in a more readable format
        print("\n📋 Detailed Historical Data:")
        for idx, row in historical.iterrows():
            print(f"  {row['ds'].strftime('%Y-%m-%d')}: RM{row['y']:,.0f}")
    
            if len(historical) < 6:
                return f"⚠️ Not enough reference data (need at least 6 months, found {len(historical)}). Please select a reference product with more history."
    
     
        # Take only first 6 months for LSTM training
        first_6_months_df = historical.head(6)
        ref_series = first_6_months_df['y'].values.reshape(-1, 1)
    
        print(f"📊 Using LSTM to learn patterns from first 6 months of reference product")
        print(f"Training data: {first_6_months_df['y'].values}")

        # Get base value for scaling (first month)
        base_value = max(1.0, ref_series[0][0])
        print(f"Base value: {base_value}")
    
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0.1, 1.0))
        scaled_data = scaler.fit_transform(ref_series)

    # Create LSTM sequences - with only 6 data points, use smaller sequence length
        SEQ_LEN = min(3, len(scaled_data) - 1)  # Use 3-month sequences for 6-month data
        print(f"Using sequence length: {SEQ_LEN}")
        
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i + seq_len])
                y.append(data[i + seq_len])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(scaled_data, SEQ_LEN)

        # Safety check
        if len(X_train) == 0:
            return "⚠️ Not enough data to build a reliable forecast model. Please use a reference product with more history."
        
        print(f"Created {len(X_train)} training sequences")

        # Train LSTM model with early stopping
        model = Sequential([
        LSTM(32, activation='relu', input_shape=(SEQ_LEN, 1), return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
        model.compile(optimizer='adam', loss='mse')
    
        
        # Train with more epochs since we have less data
        early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=300, verbose=0, callbacks=[early_stopping])
    
        print(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")


        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_unscaled = scaler.inverse_transform(y_pred_train)
        y_actual_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1))
    
        # Calculate training accuracy
        denominator = np.maximum(1.0, np.abs(y_actual_unscaled))
        mape = np.mean(np.abs((y_actual_unscaled - y_pred_unscaled) / denominator)) * 100
        accuracy = max(0, min(100, 100 - mape))
        print(f"📈 Model Training Accuracy: {accuracy:.2f}% (MAPE: {mape:.2f}%)")
        # Calculate accuracy on test data if available
        confidence_level = "high" if accuracy > 85 else "moderate" if accuracy > 70 else "low"
        accuracy_info = f"<br><strong>Model Accuracy:</strong> {accuracy:.2f}% ({confidence_level} confidence on first 6 months pattern)<br>"

        input_seq = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        forecast_values = []
    
        for month_idx in range(6):
            pred = model.predict(input_seq, verbose=0)[0][0]
            forecast_values.append(pred)
        
        # Update sequence for next prediction
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        # Convert back to original scale
        forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()

        # Format output with enhanced information
        total_forecast = np.sum(forecast_values)
        monthly_breakdown = ""
        for i in range(6):
            value = forecast_values[i]
            monthly_breakdown += f"- Month {i+1}: RM{value:,.0f}<br>"



      
        monthly_avg = total_forecast / 6
        min_forecast = np.min(forecast_values)
        max_forecast = np.max(forecast_values)
        volatility = (np.std(forecast_values) / monthly_avg) * 100

       
       # Show the reference pattern for comparison
        reference_pattern = ""
        for i, value in enumerate(first_6_months_df['y'].values):
            month_name = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"][i]
            reference_pattern += f"- {month_name}: RM{value:,.0f}<br>"

        return (
        f"<h3>🎯 New Product Sales Forecast (LSTM Trained on First 6 Months)</h3>"
        f"<p>This forecast uses an LSTM model trained specifically on the <strong>first 6 months</strong> "
        f"of the reference product to learn early-stage sales patterns.</p>"
        
        f"<h4>Reference Product's First 6 Months:</h4>"
        f"{reference_pattern}"
        
        f"<h4>New Product Forecast Summary:</h4>"
        f"<p><strong>Monthly Average:</strong> RM{monthly_avg:,.0f}</p>"
        f"<p><strong>Total 6-Month Revenue:</strong> RM{total_forecast:,.0f}</p>"
        f"<p><strong>Range:</strong> RM{min_forecast:,.0f} to RM{max_forecast:,.0f}</p>"
        f"<p><strong>Volatility:</strong> {volatility:.1f}% (month-to-month variation)</p>"
        
        f"{accuracy_info}"
        
        f"<h4>Monthly Forecast Breakdown:</h4>"
        f"{monthly_breakdown}"
        
        f"<p><em>Note: This forecast is based on learning patterns from only the first 6 months of the reference product, "
        f"making it ideal for predicting new product launch performance. The model has learned the early adoption curve "
        f"and growth patterns specific to the product's initial market entry phase.</em></p>"
        )


    recent_data = forecast_df[['ds', 'yhat', 'adjusted_yhat']].tail(min(months, len(forecast_df)))

    # Calculate metrics
    prophet_avg = recent_data['yhat'].mean()
    lstm_avg = recent_data['adjusted_yhat'].mean()
    difference = lstm_avg - prophet_avg
    pct_difference = (difference / prophet_avg) * 100 if prophet_avg != 0 else 0

    # Determine trend message
    if abs(pct_difference) < 5:
        adjustment_msg = "The LSTM adjustment made minimal changes to the Prophet forecast."
    else:
        direction = "increased" if difference > 0 else "decreased"
        adjustment_msg = (
            f"The LSTM adjustment {direction} the forecast by {abs(pct_difference):.1f}% "
            f"compared to Prophet alone, suggesting it detected meaningful monthly patterns."
        )

    # Main summary
    summary = f"""
    <h3>📊Forecast Summary</h3>
    <p><strong>Forecast Avg:</strong> RM{lstm_avg:,.0f} per month</p>
    
    <h4>Key Statistics:</h4>
    <p>📈 Highest Forecasted Month: RM{forecast_df['adjusted_yhat'].max():,.0f}</p>
    <p>📉 Lowest Forecasted Month: RM{forecast_df['adjusted_yhat'].min():,.0f}</p>
    <p>📊 Average Monthly Sales: RM{lstm_avg:,.0f}</p>
    """.strip()

    if len(recent_data) == 1:
        summary += "<br><em>Note: This forecast includes only one month of data. Interpret with caution.</em>"

   
    summary += "<br><br><strong>📅 Forecasted Sales by Month:</strong><br>"
    monthly_view = forecast_df[['ds', 'adjusted_yhat']].tail(months)
    for _, row in monthly_view.iterrows():
        date_str = pd.to_datetime(row['ds']).strftime('%B %Y')
        summary += f"- {date_str}: RM{row['adjusted_yhat']:,.0f}<br>"

    return summary


def get_chart_base64(image_path):
    with open(image_path, "rb") as img_file:
        b64_data = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/png;base64,{b64_data}"

#forecasting end

def clean_response(text):
    match = re.search(r'([^.?!]*[.?!])', text)
    return match.group(0).strip() if match else text.strip()

def get_database_schema():
    try:
        conn = pyodbc.connect(
            r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=EBuilder;Trusted_Connection=yes;"
        )
        cursor = conn.cursor()

        schema = {}

        # Get columns for the view var_trx_sales_analysis_by_ar
        cursor.execute("SELECT * FROM var_trx_sales_analysis_by_ar")
        schema['var_trx_sales_analysis_by_ar'] = {
            "description": "Sales Analysis View",
            "columns": [desc[0] for desc in cursor.description]
        }

        # Get columns for the table ar_trx
        cursor.execute("SELECT * FROM AR_TRX")
        schema['AR_TRX'] = {
            "description": "AR Transactions Table",
            "columns": [desc[0] for desc in cursor.description]
        }

        # Get columns for the table AR_MTN
        cursor.execute("SELECT * FROM AR_MTN")
        schema['AR_MTN'] = {
            "description": "Customer Maintenance Table",
            "columns": [desc[0] for desc in cursor.description]
        }
        
        # Get columns for the table STK_MTN
        cursor.execute("SELECT * FROM STK_MTN")
        schema['STK_MTN'] = {
            "description": "Stock Master Table",
            "columns": [desc[0] for desc in cursor.description]
        }

        # Get columns for the table AR_TRX_D
        cursor.execute("SELECT * FROM AR_TRX_D")        
        schema['AR_TRX_D'] = {
            "description": "AR Transaction Details Table",
            "columns": [desc[0] for desc in cursor.description]
        }

        # Get columns for the table SM_MTN_CAT1
        cursor.execute("SELECT Cat1ID,Cat1Desc FROM SM_MTN_CAT1")
        schema['SM_MTN_CAT1'] = {
            "description": "Stock Category 1 Table",
            "columns": [desc[0] for desc in cursor.description]
        }

        # Get columns for the table SM_MTN_CAT2
        cursor.execute("SELECT Cat2ID,Cat2Desc FROM SM_MTN_CAT2")
        schema['SM_MTN_CAT2'] = {
            "description": "Stock Category 2 Table",
            "columns": [desc[0] for desc in cursor.description]
        }
        return schema
    

    except pyodbc.Error as e:
        print(f"Database schema error: {str(e)}")
        return {}
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass


def extract_query_details_from_llm(user_input):
    session_id = request.sid
    history = user_histories[session_id][-6:] if session_id in user_histories else []
    
    # Create context from conversation history
    history_context = ""
    if history:
        history_context = "Previous conversation:\n" + "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in history
        ])
    
    schema = get_database_schema()
   
    schema_text = "\n".join(
        f"{table}: {', '.join(info['columns'])}" for table, info in schema.items()
    )

    # Add a relationship hint for the LLM
  
    relationship_hint = (
    "Relationship: AR_TRX_D.TrxStkId = STK_MTN.StkId\n"
    "If the user asks for stock names with sales, join AR_TRX_D and STK_MTN on TrxStkId = StkId\n "
    "The sales amount column is AR_TRX_D.TrxAmt.\n"
    "Do NOT filter by TrxType unless the user explicitly asks.\n"
    "The stock category column in var_trx_sales_analysis_by_ar is StkCat3.\n"
    "The stock ID column in STK_MTN is StkId.\n"
    "The stock name column in STK_MTN is StkName1 + StkName2.\n"

    "The quantity column in var_trx_sales_analysis_by_ar is TrxQty.\n"
    "The product name column is TrxStkDesc1.\n"
    "The product ID column in AR_TRX_D is TrxStkId.\n"
    "The product amount column in var_trx_sales_analysis_by_ar is ItemAmt\n"
    "The product code column in STK_MTN is StkCode.\n"
    "The product category columns in STK_MTN are StkCat1 and StkCat2.\n"
    "the product category description columns in STK_MTN_CAT1 is Cat1Desc and STK_MTN_CAT2 is Cat2Desc\n"
   

    "The sales amount column in var_trx_sales_analysis_by_ar is TrxAmt.\n"
    "The types of stock in STK_MTN are StkCat3.(e.g.,how many stock of BASEUS in the warehouse) \n"
    "Can get the top 5 sales customers from var_trx_sales_analysis_by_ar by sum ItemAmt\n"
    "Can get what products were sold to a customer from var_trx_sales_analysis_by_ar and sum ItemAmt\n"
    "The Transaction number column in AR_TRX_D is TrxNo.\n"
    
    "the customer region column in AR_MTN is ArCat5.\n"
    "The customer short name column in var_trx_sales_analysis_by_ar is ArNameS and full name is ArName1. (e.g. short name OEL JH)\n"
    "The customer code column in AR_MTN and var_trx_sales_analysis_by_ar is ArCode(e.g., 3000/C0002,3000/I0004).\n"
    "The customer ID column in AR_MTN and  is CoID.\n"

    "The salesman name column in var_trx_sales_analysis_by_ar is SmanDesc.\n"
    "The sales amount of salesman column in var_trx_sales_analysis_by_ar is ItemAmt.\n"
   
    "when the ask for top 5 customers by sales, no need group with SmanDesc.Instead, show SmanDesc using an aggregate function like MAX(v.SmanDesc) to avoid violating GROUP BY rules.\n"
    "The category description columns are SM_MTN_CAT1.Cat1Desc and SM_MTN_CAT2.Cat2Desc.\n"
    "Relationships: STK_MTN.StkCat1 = SM_MTN_CAT1.Cat1ID AND STK_MTN.StkCat2 = SM_MTN_CAT2.Cat2ID AND s.CoID = c1.CoID and s.CoID = c2.CoID.\n"
    "**IMPORTANT**: When the user asks for sales, ALWAYS use AR_TRX_D or var_trx_sales_analysis_by_ar.\n"
    "**IMPORTANT**: Ignore TrxType unless the user asks for a specific type (e.g., only 'INV')\n."
    "**IMPORTANT**: When aggregating sales data (e.g., top 5 products), always GROUP BY product ID (TrxStkID). Do not group by product name or description alone, as multiple products may share similar descriptions. Join with STK_MTN to retrieve product names (StkNameS) and codes (StkCode) after grouping by ID.\n"
   
    
    )


    prompt = f"""
You are a SQL Server expert. Based on this database schema and relationships:

{schema_text}

{relationship_hint}

🚨 CRITICAL RULE - READ CAREFULLY:
If a column such as 'TrxStkID' is needed for JOIN or GROUP BY, you may use it internally,but you MUST NOT include it in the SELECT clause unless the user explicitly requests it.

Your task:
First, read the full conversation below to determine context:

{history_context}

Convert the following user request into a valid Microsoft SQL Server query.
Only use column names that appear in the schema above.

Your task:
Determine whether the user request requires a SQL query. If the request is general (e.g., about business context or product explanation), respond with: "This is a general question and does not require a SQL query."

If the user is referring to something mentioned in previous messages, use that context to generate the appropriate SQL query.

If a query is needed:
- Use ONLY the exact column names provided in the schema. Do NOT invent column names.
- Return ONLY the SQL query, no commentary or explanation.
- Use single quotes for string literals (e.g., WHERE x = 'value').
- Use table aliases to keep queries concise.
- The output must be a valid SQL Server query ready to run in SSMS.
- When the user asks "how many stock", assume it means counting records (e.g., COUNT(*)), not summing StkWeigth1 or any weight-related field.
- Do NOT use the 'StkWeigth1' column unless the user explicitly requests weight-based calculations.\n"
- Do NOT generate forecasting SQL queries (e.g., time series, Prophet, LSTM, forecasting). Focus ONLY on retrieving and aggregating historical data from the database.


Current user request: "{user_input}"
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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"
            },
            json={
                "model": "deepseek/deepseek-chat-v3-0324",
              
                "messages": [
                    {"role": "system", "content": "You convert questions to SQL using the schema and conversation history."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 800,
            },
        )
        data = response.json()

        if not data.get("choices"):
            print(f"Unexpected LLM response: {data}")
            return {}

        content = data["choices"][0]["message"]["content"].strip()
        # Remove code fences
        if content.startswith("```sql"):
            content = content.replace("```sql", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        content = content.replace("`", "").strip()

        # Only keep the first SQL statement (stop at first semicolon or newline after SELECT)
        # This helps if the LLM returns explanations or extra text
        lines = content.splitlines()
        sql_lines = []
        for line in lines:
            if line.strip().lower().startswith("select") or sql_lines:
                sql_lines.append(line)
                if ";" in line:
                    break
        content = "\n".join(sql_lines).strip()

        if content.endswith(("UNION", "SELECT", "FROM", "WHERE", "AND", "OR", ".", ",")):
            print("LLM returned an incomplete query. Skipping execution.")
            return {}
        
        if content.lower().startswith("this is a general question"):
            print("LLM determined this is a general question, not requiring SQL.")
            return {}
            
        return {"sql_query": content}

    except Exception as e:
        print(f"Error in LLM parsing: {str(e)}")
        return {}



def generate_llm_response(user_input):
    try:
        session_id = request.sid
        history = user_histories[session_id][-6:] if session_id in user_histories else []
        
        # Create context from conversation history
        history_context = ""
        if history:
            history_context = "Previous conversation:\n" + "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in history
            ])
       
        
       
        system_content = (
        "You are a helpful assistant for an ERP system. "
        "First, review the conversation history to understand context. "
        "Then provide a helpful response to the current question. "
        "Format your response using HTML for better readability. "
        "Use appropriate HTML tags like <p>, <h3>, etc. "
        "Avoid using bullet points and <ul>/<li> tags - use paragraphs instead. "
        "Do not include <html>, <body>, or other document-level tags."
         )


        prompt = f"{history_context}\n\nCurrent question: {user_input}"

        try:
            response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
                headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
        },
                json={
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 800
        }
    )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
                
                # Wrap in div if not already using HTML
            if not ("<p>" in content or "<h" in content or "<ul>" in content or "<div>" in content):
                content = f"<div>{content}</div>"
                    
            return content
        except Exception as e:
                print(f"Error calling OpenRouter API: {str(e)}")
                return "I'm having trouble responding. Could you try again?"
    except Exception as e:
                print(f"Error in generate_llm_response: {str(e)}")
                return "I encountered an error processing your request. Please try again."
    except Exception as e:
                print(f"Error in generate_llm_response: {str(e)}")
                return "I encountered an error processing your request. Please try again."
    


def format_rows_for_display(rows):
    formatted = []
    for row in rows:
        formatted_row = []
        for value in row:
            if isinstance(value, (float, int, decimal.Decimal)):
                # Always show as 2 decimal places with commas
                formatted_row.append(locale.format_string('%.2f', float(value), grouping=True))
            else:
                formatted_row.append(value)
        formatted.append(formatted_row)
    return formatted




def clean_forecast_data(df, value_col='y', date_col='ds'):
    """
    Enhanced cleaning of forecast data with multiple strategies for negative values.
    Provides diagnostic information about the data quality.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with time series data
        value_col (str): Column name containing values to clean
        date_col (str): Column name containing dates
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
        dict: Diagnostic information about the cleaning process
    """

    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    diagnostics = {}
    
    # Basic statistics before cleaning
    diagnostics['original_negative_count'] = len(cleaned_df[cleaned_df[value_col] < 0])
    diagnostics['original_negative_percentage'] = diagnostics['original_negative_count'] / len(cleaned_df)
    diagnostics['original_mean'] = cleaned_df[value_col].mean()
    
    # Strategy 1: Only replace negatives when they're likely errors
    # Calculate rolling median to identify outliers
    window_size = min(12, len(cleaned_df)//2)  # Adaptive window size
    rolling_median = cleaned_df[value_col].rolling(window=window_size, min_periods=1).median()
    std_dev = cleaned_df[value_col].std()
    
    # Identify values that are both negative AND significant outliers
    outlier_mask = (cleaned_df[value_col] < 0) & \
                  (abs(cleaned_df[value_col] - rolling_median) > 3 * std_dev)
    
    # Replace only the most extreme negative values
    cleaned_df.loc[outlier_mask, value_col] = np.nan
    
    # Strategy 2: Interpolate missing values (including the ones we just set to NaN)
    cleaned_df[value_col] = cleaned_df[value_col].interpolate(method='time')
    
    # Strategy 3: For any remaining negatives, use exponential transformation
    if (cleaned_df[value_col] < 0).any():
        min_val = cleaned_df[value_col].min()
        if min_val < 0:
            # Apply softplus transformation to handle negatives while preserving scale
            cleaned_df[value_col] = np.log(1 + np.exp(cleaned_df[value_col] - min_val)) + min_val
    
    # Post-cleaning diagnostics
    diagnostics['final_negative_count'] = len(cleaned_df[cleaned_df[value_col] < 0])
    diagnostics['values_modified'] = len(df) - (df[value_col] == cleaned_df[value_col]).sum()
    
    return cleaned_df, diagnostics
    


def is_reference_question(user_input):
    """
    Determines if a user question is asking about using a product as a reference
    for another product by sending the query to an LLM.
    """
    try:
        prompt = f"""
        Determine if this question is asking about using one product's data as a reference or baseline 
        for another product (like a new product launch). Answer only 'yes' or 'no'.

        Question: "{user_input}"
        """

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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"
            },
            json={
                "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                "messages": [
                    {"role": "system", "content": "You determine if questions are about using product data as reference."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 10,
            },
        )

        res_json = response.json()
        if "choices" in res_json and len(res_json["choices"]) > 0:
            answer = res_json["choices"][0]["message"]["content"].strip().lower()
            return 'yes' in answer
        else:
            raise ValueError(f"Unexpected response: {res_json}")

    except Exception as e:
        logging.error("Error in reference detection: %s", str(e), exc_info=True)  # Log stack trace
        # Fall back to keyword matching if LLM fails
        keywords = [
            'reference', 'as reference', 'based on', 'similar product', 'compare to', 'to estimate', 'refer',
            'similar to', 'like', 'comparable to', 'same as', 'equivalent to', 'analogous to',
            'benchmark', 'model after', 'pattern after', 'guide', 'template', 'proxy for',
            'substitute', 'replacement', 'alternative', 'stand-in', 'baseline', 'starting point',
            'new product', 'launching', 'introduction', 'launching soon', 'use as example'
        ]
        return any(kw in user_input.lower() for kw in keywords)
    

def prepare_table_analysis_context(user_input, sql_query, table_data, columns, row_count):
    """
    Prepare structured context for LLM analysis of table results
    """
    # Sample data for analysis (limit to first 10 rows to avoid token limits)
    sample_data = table_data[:10] if len(table_data) > 10 else table_data
    
    # Calculate basic statistics if numeric columns exist
    numeric_stats = {}
    for col in columns:
        try:
            values = [row[col] for row in table_data if row[col] is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                numeric_stats[col] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        except:
            continue
    
    context = {
        'user_question': user_input,
        'sql_query': sql_query,
        'total_rows': row_count,
        'columns': columns,
        'sample_data': sample_data,
        'numeric_statistics': numeric_stats,
        'data_types': {col: type(table_data[0][col]).__name__ if table_data else 'unknown' for col in columns}
    }
    
    return context


def get_llm_table_analysis(context):
    """
    Send table context to LLM for intelligent analysis and insights



    """
    try:
        # Construct prompt for LLM analysis
        analysis_prompt = f"""
        You are a business intelligence analyst. A user asked: "{context['user_question']}"
        
        The following SQL query was executed:
        ```sql
        {context['sql_query']}
        ```
        
        Results Summary:
        - Total rows returned: {context['total_rows']}
        - Columns: {', '.join(context['columns'])}
        
        Sample Data (first 10 rows):
        {format_sample_data_for_llm(context['sample_data'])}
        
        Numeric Statistics:
        {format_numeric_stats_for_llm(context['numeric_statistics'])}
        
        Please provide:
        1. A clear interpretation of what this data shows
        2. Key insights and patterns you notice
        3. Business implications or recommendations
       
        
        
        
       Please provide the following analysis, formatted directly in clean HTML (without wrapping it in Markdown or code blocks):

<h2>1. Interpretation</h2>
<ul>
  <li>Describe what the data shows in context.</li>
</ul>

<h2>2. Key Insights and Patterns</h2>
<ul>
  <li>Highlight important trends, distributions, or standout values.</li>
</ul>

<h2>3. Business Implications and Recommendations</h2>
<ul>
  <li>Suggest actionable steps based on the data insights.</li>
</ul>



Keep your response concise, insightful, and ready for embedding in a BI dashboard.
"""
        
        # Call your LLM API (replace with your actual LLM integration)
        response = call_llm_api(analysis_prompt)
        
        return response
        
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return f"<p>✅ Query executed successfully returning {context['total_rows']} rows. Raw data displayed above.</p>"


def format_sample_data_for_llm(sample_data):
    """Format sample data for LLM consumption"""
    if not sample_data:
        return "No data available"
    
    formatted = []
    for i, row in enumerate(sample_data, 1):
        row_str = f"Row {i}: " + ", ".join([f"{k}: {v}" for k, v in row.items()])
        formatted.append(row_str)
    
    return "\n".join(formatted)


def format_numeric_stats_for_llm(numeric_stats):
    """Format numeric statistics for LLM consumption"""
    if not numeric_stats:
        return "No numeric columns found"
    
    formatted = []
    for col, stats in numeric_stats.items():
        stats_str = f"{col}: Min={stats['min']}, Max={stats['max']}, Avg={stats['avg']:.2f}, Count={stats['count']}"
        formatted.append(stats_str)
    
    return "\n".join(formatted)



def call_llm_api(prompt):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324",
                    "messages": [
                        {"role": "system", "content": "You are a business intelligence analyst providing insights."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            return "Unable to process the request. Please try again later."


def database_forecast_question(user_input):
    """
    Determines if the user question is asking for a database query that requires forecasting.
    Returns True if the question asks for top sales products for a customer and then a forecast.
    """
    user_input_lower = user_input.lower()
    # Detect pattern: "give me ArNameS OEL JH top 5 sales product then get best sales stock code then do forecast 6 month"
    # or similar requests for top products and forecasting
    if (
        ("top 5 sales product" in user_input_lower or "top sales product" in user_input_lower)
        and ("forecast" in user_input_lower or "do forecast" in user_input_lower)
        and ("arname" in user_input_lower or "customer" in user_input_lower or "oel jh" in user_input_lower)
    ):
        return True
    # Fallback to LLM for other cases
    try:
        prompt = f"""
        Does this question require a SQL query to get top sales products for a customer and then forecast sales for the best product?
        Answer only 'yes' or 'no'.

        Question: "{user_input}"
        """

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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"
            },
            json={
                "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                "messages": [
                    {"role": "system", "content": "You determine if questions require top product SQL and forecasting."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 10,
            },
        )

        res_json = response.json()
        if "choices" in res_json and len(res_json["choices"]) > 0:
            answer = res_json["choices"][0]["message"]["content"].strip().lower()
            return 'yes' in answer
        else:
            return False

    except Exception as e:
        logging.error("Error in forecast detection: %s", str(e), exc_info=True)
        return False


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_message')
def handle_user_message(data):
    user_input = data['message']
    session_id = request.sid
    
    user_histories[session_id].append({"role": "user", "content": user_input})

    def send_response(response):
        socketio.emit('assistant_response', {'text': response}, room=session_id)
        user_histories[session_id].append({"role": "assistant", "content": response})

    if is_forecast_question(user_input):
        print("🚀 Starting forecasting process...")
        send_response("🔍 Processing your forecast request...")
        
        try:

            history = user_histories[session_id][-6:]  # last 6 messages
            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            messages.append({"role": "user", "content": user_input})


            if database_forecast_question(user_input):
                send_response("🔍 Analyzing your query for database execution...")
                query_details = extract_query_details_from_llm(user_input)
                sql_query = query_details.get("sql_query")

                print(f"📝 Extracted SQL query: {sql_query}")

                try:
                    conn = pyodbc.connect(
                    r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=EBuilder;Trusted_Connection=yes;"
                    )
                    cursor = conn.cursor()
                    cursor.execute(sql_query)
                    rows = cursor.fetchall()
                    columns = [column[0] for column in cursor.description]
                    print('rows',rows)
                  
                    try:
    
                        print("📌 Sample row:", rows[0])
                        print("📌 Columns:", columns)
                        print("📌 Row length:", len(rows[0]), "Column length:", len(columns))

                        print("Type of rows[0]:", type(rows))
                        print("🔎 Type of rows[0]:", type(rows[0]))
                        print("🔎 Type of rows[0][0]:", type(rows[0][0]))

                        rows = [tuple(r) for r in rows]

  
                        if len(rows[0]) == len(columns):
                             print("✅ Row length matches column length. Creating DataFrame...")
                             df_all = pd.DataFrame(rows, columns=columns)
                       
                        else:
                            print("⚠️ Unexpected shape. Applying fallback columns.")
                            df_all = pd.DataFrame(rows)
                            df_all.columns = [f"col{i+1}" for i in range(df_all.shape[1])]


                    except Exception as e:
                        print(f"❌ Error building DataFrame: {e}")
                        df_all = pd.DataFrame()

                             

                    if not rows:
                       send_response("No results found for that query.")
                    else:
                       columns = [column[0] for column in cursor.description]
                    
                    # Convert results to a more structured format for LLM analysis
                    table_data = []
                    for row in rows:
                        table_data.append(dict(zip(columns, row)))
                    
                    # Create formatted table for display
                    formatted_table = tabulate(format_rows_for_display(rows), headers=columns, tablefmt="grid")
                    
                    # **NEW: Enhanced LLM Analysis**
                    print("🧠 Sending data to LLM for intelligent analysis...")
                    send_response("🔍 Analyzing results with AI...")
                    
                    # Prepare data context for LLM
                    analysis_context = prepare_table_analysis_context(
                        user_input=user_input,
                        sql_query=sql_query,
                        table_data=table_data,
                        columns=columns,
                        row_count=len(rows)
                    )
                    history = user_histories[session_id][-6:]  # last 6 messages
                    messages = [{"role": m["role"], "content": m["content"]} for m in history]
                    messages.append({"role": "user", "content": user_input})
                    # Get intelligent analysis from LLM
                    llm_analysis = get_llm_table_analysis(analysis_context)
                    
                    # Combine table display with intelligent analysis
                    enhanced_response = f"""
                    <div style="margin-bottom: 20px;">
                        <h3>📊 Query Results:</h3>
                        <pre>{formatted_table}</pre>
                    </div>
                    
                    <div style="margin-top: 20px; padding: 30px; background-color: #f8f9fa;padding-left: 20px;">
                        <h3>Analysis:</h3>
                        {llm_analysis}
                    </div>
                    """
                    
                    send_response(enhanced_response)
                    
                 

                  

                    if 'StkCode' not in df_all.columns:
                        print("No StkCode column found in results, skipping stock code extraction.")
                    else:
                        send_response("📦 Analyzing product codes for forecasting...")
                        top_5_codes = df_all['StkCode'].unique().tolist()
                        combined_response = ""
                        for code in top_5_codes:
                            try:
                                send_response(f"📦 Processing data for product code: {code}...")

                                sql_forecast = generate_product_forecast_sql(user_input, code)
                                if not sql_forecast:
                                    combined_response += f"❌ Could not generate query for product code {code}.\n"
                                    continue

                              

                                df = fetch_product_forecast_data(sql_forecast)
                                if df is None or df.empty:
                                    combined_response += f"❌ No historical data found for product code {code}. Cannot generate forecast.\n"
                                    continue

                                df['y'] = df['y'].clip(lower=0)

                                if len(df) < 10:
                                    combined_response += f"❌ Not enough data points for product code {code}. Minimum 10 required.\n"
                                    continue

                                forecast_period = extract_forecast_period(user_input)

                                forecast_result = hybrid_forecast(df,periods=forecast_period)
                                if not forecast_result:
                                    combined_response += f"<p>❌ Forecast failed for <b>{code}</b></p>"
                                    continue

                               
                                
                                plot_path = save_forecast_plot(
                                    forecast_result['historical_data'],
                                    forecast_result['adjusted_forecast'],
                                    forecast_result['prophet_model']
                                )

                                chart_data = get_chart_base64(plot_path)

                                summary = generate_hybrid_forecast_summary(
                                    forecast_result['adjusted_forecast'],
                                    forecast_period,
                                    detailed=True if forecast_period == 12 else False,
                                    reference=False,
                                    historical_df=df.copy()
                                )



                                combined_response += f"""
                                <div style="margin-bottom:30px;">
                                    <p>{summary}</p>
                                    <h3>📦 Forecast for: {code}</h3>
                                    
                                    <img src="{chart_data}" alt="Forecast Chart" style="max-width:100%;">
                                </div>
                                """
                                
                            except Exception as e:
                                error_msg = f"❌ Forecast error: {str(e)}"
                                print(error_msg)
                                import traceback
                                traceback.print_exc()
                                combined_response += f"{error_msg}\n\nPlease try again or contact support."

                        send_response("✅ Forecast complete")
                        send_response(combined_response)


                    
          
               
                    exit(0)  # Exit after processing database query

                except Exception as e:
                    error_msg = f"Error executing query: {str(e)}"
                    print(error_msg)
                    
                finally:
                    try:
                       cursor.close()
                       conn.close()
                    except:
                       pass

            
            forecast_period = extract_forecast_period(user_input)
            print(f"📅 Forecast pergenerate_product_forecast_sqliod: {forecast_period} months")
           
            if forecast_period > 12:
              warning_msg = (
                  f"⚠️ You requested a forecast for {forecast_period} months. "
                  f"Note: Forecasts beyond 12 months are less accurate due to increasing uncertainty. "
                  f"Please interpret results with caution."
              )
              send_response(warning_msg)

            # Generate appropriate SQL based on whether it's product-specific
            if is_product_forecast_question(user_input):
                print("📦 Generating product-specific forecast generate_product_forecast_sqlt...")
                #sql_query = generate_product_forecast_sql(user_input)
                sql_query = reference_product_sql(user_input)
                print("sql_query_2:",sql_query)
                
               
                
                if not sql_query:
                    send_response("❌ Could not generate query for product forecast.")
                    return
                    
                print(f"🔍 Generated SQL: {sql_query[:100]}...")
                send_response("📊 Fetching product data...")
                df = fetch_product_forecast_data(sql_query)
                
            else:
                print("📈 Generating general sales forecast...")
                send_response("📊 Fetching sales data...")
                df = fetch_forecast_data()

             
            
            # Data validation
            if df is None:
                send_response(f"❌ No data connection established .")
                return
                
            if df.empty:
                send_response(f"❌ No historical data found. Cannot generate forecast.")
                return
            
            
            
            print(f"✅ Data loaded: {len(df)} rows")
            print(f"📊 Date range: {df['ds'].min()} to {df['ds'].max()}")
            print(f"💰 Sales range: {df['y'].min():.2f} to {df['y'].max():.2f}")
            

            # Step 2: Data Cleaning (in-place modification)
            send_response("🧹 Cleaning data...")
            original_row_count = len(df)
        
        # Clean the data while keeping the same dataframe variable
            cleaning_report = {}
            cleaning_report['original_negative_count'] = len(df[df['y'] < 0])
        
        # Perform cleaning directly on df
            df['y'] = df['y'].clip(lower=0)  # Replace negatives with 0
        
        # Optional: Advanced cleaning could go here
        # df['y'] = np.where(df['y'] < 0, df['y'].rolling(3, min_periods=1).mean(), df['y'])
        
            cleaning_report['values_modified'] = cleaning_report['original_negative_count']
        
        # Cleaning Summary
            print(f"✅ Data cleaned - Removed {cleaning_report['original_negative_count']} negatives")
            send_response(f"✅ Cleaned {len(df)} data points (removed {cleaning_report['original_negative_count']} negatives)")
        
        # Data Quality Check
            print(f"📊 Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
            print(f"💰 Value range: {df['y'].min():,.2f} to {df['y'].max():,.2f}")
            print("Raw data before processing:")
            print(df.to_string())
            # Check for minimum data requirements
            if len(df) < 10:
                send_response(f"⚠️ Only {len(df)} data points available. Need at least 10 for reliable forecasting.")
                return
            
            send_response("🧠 Running Prophet + LSTM hybrid forecast...")
            
            # Run hybrid forecast with detailed logging
            print("🔮 Starting hybrid forecast...")
            forecast_result = hybrid_forecast(df, periods=forecast_period)
            
            if forecast_result is None:
                send_response("❌ Forecast generation failed.")
                return
            
            print("📊 Generating forecast plot...")
            # Save plot with both forecasts
            plot_path = save_forecast_plot(
                forecast_result['historical_data'],
                forecast_result['adjusted_forecast'],
                forecast_result['prophet_model']
            )
            
            print("📝 Generating summary...")
            reference_flag = is_reference_question(user_input)
            print(f"🔍 Reference question detected: {reference_flag}")

            #print("First 5 rows of df:")
            #print(df.head())

            reference_df = df.copy()
            reference_df['y'] = reference_df['y_original'] 


            summary = generate_hybrid_forecast_summary(
               forecast_result['adjusted_forecast'], 
               forecast_period, 
               detailed=True if forecast_period == 12 else False,
               reference=reference_flag,
               historical_df=reference_df
               )

            detail_summary = generate_forecast_summary(user_input,summary)
            
            # Only generate chart if not a reference question
            if not reference_flag:
                print("🖼️ Preparing chart display...")
                chart_data = get_chart_base64(plot_path)
                
                # Send response with chart
                response_html = f"""
                <p>{summary}</p>
                
                <h3>🖼️ Forecast Chart:</h3>
                <img src="{chart_data}" alt="Forecast Chart" style="max-width:100%;">
                <p>{detail_summary}</p>
                <p><small>Note: Red dashed line shows LSTM-adjusted forecast</small></p>
                """
            else:
                # Send response without chart for reference questions
                response_html = f"""
                <p>{summary}</p>
                
                <p>{detail_summary}</p>
                """
            
            print("✅ Forecast completed successfully!")
            send_response(response_html)
            
        except Exception as e:
            error_msg = f"❌ Forecast error: {str(e)}"
            print(error_msg)
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            send_response(f"{error_msg}\n\nPlease try again or contact support if the issue persists.")
        
    else:
        # Handle regular database queries with enhanced LLM analysis
        history = user_histories[session_id][-6:]  # last 6 messages
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": user_input})

       
        
        query_details = extract_query_details_from_llm(user_input)
        sql_query = query_details.get("sql_query")
        print(f"Extracted SQL query: {sql_query}")

        if sql_query:
            send_response("Let me analyze that data for you...")
            
            try:
                conn = pyodbc.connect(
                    r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=EBuilder;Trusted_Connection=yes;"
                )
                cursor = conn.cursor()
                cursor.execute(sql_query)
                rows = cursor.fetchall()

                if not rows:
                    send_response("No results found for that query.")
                else:
                    columns = [column[0] for column in cursor.description]
                    
                    # Convert results to a more structured format for LLM analysis
                    table_data = []
                    for row in rows:
                        table_data.append(dict(zip(columns, row)))
                    
                    # Create formatted table for display
                    formatted_table = tabulate(format_rows_for_display(rows), headers=columns, tablefmt="grid")
                    
                    # **NEW: Enhanced LLM Analysis**
                    print("🧠 Sending data to LLM for intelligent analysis...")
                    send_response("🔍 Analyzing results with AI...")
                    
                    # Prepare data context for LLM
                    analysis_context = prepare_table_analysis_context(
                        user_input=user_input,
                        sql_query=sql_query,
                        table_data=table_data,
                        columns=columns,
                        row_count=len(rows)
                    )
                    history = user_histories[session_id][-6:]  # last 6 messages
                    messages = [{"role": m["role"], "content": m["content"]} for m in history]
                    messages.append({"role": "user", "content": user_input})
                    # Get intelligent analysis from LLM
                    llm_analysis = get_llm_table_analysis(analysis_context)
                    
                    # Combine table display with intelligent analysis
                    enhanced_response = f"""
                    <div style="margin-bottom: 20px;">
                        <h3>📊 Query Results:</h3>
                        <pre>{formatted_table}</pre>
                    </div>
                    
                    <div style="margin-top: 20px; padding: 30px; background-color: #f8f9fa;padding-left: 20px;">
                        <h3>Analysis:</h3>
                        {llm_analysis}
                    </div>
                    """
                    
                    send_response(enhanced_response)

            except Exception as e:
                error_msg = f"Error executing query: {str(e)}"
                print(error_msg)
                send_response(error_msg)
            finally:
                try:
                    cursor.close()
                    conn.close()
                except:
                    pass
        else:
            
            # No SQL query could be generated, so fall back to LLM for a natural language answer
            response = generate_llm_response(user_input)
            send_response(response)


        
        

if __name__ == '__main__':
    # Print out the database schema (tables and columns)
    #schema = get_database_schema()
    #for table, info in schema.items():
    #     print(f"Table: {table}")
     #    print("Columns:")
     #    for col in info['columns']:
     #        print(f"  - {col}")
      #   print()
    socketio.run(app, debug=True, use_reloader=False)
