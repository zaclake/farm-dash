# modules/machine_learning.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import time

def run_machine_learning_tab(ml_data, configuration):
    st.header("Machine Learning Predictions and Optimization")

    # Instructions or description of the tab
    st.write("""
    Use this tab to generate predictions and optimizations using machine learning models.
    Select your target metric and features, adjust advanced settings, and visualize the results.
    """)

    # Step 1: Target Metric Selection
    target_options = [col for col in ml_data.columns if col != 'date']
    target_display_names = [col.replace('_', ' ').title() for col in target_options]
    target_name_mapping = dict(zip(target_display_names, target_options))
    selected_target_display = st.selectbox("Select Target Metric to Predict or Optimize", target_display_names)
    selected_target = target_name_mapping[selected_target_display]

    # Step 2: Feature Selection
    feature_options = [col for col in ml_data.columns if col != 'date' and col != selected_target]
    feature_display_names = [col.replace('_', ' ').title() for col in feature_options]
    feature_name_mapping = dict(zip(feature_display_names, feature_options))
    selected_feature_display = st.multiselect("Select Features to Use in the Model", feature_display_names)
    selected_features = [feature_name_mapping[fdn] for fdn in selected_feature_display]

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # Step 3: Advanced Settings
    with st.expander("Advanced Settings"):
        st.write("Adjust hyperparameters for the XGBoost model.")
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
        max_depth = st.slider("Max Depth", min_value=1, max_value=15, value=6, step=1)
        n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=100, step=10)
    # Else, use default values

    # Prepare data
    X = ml_data[selected_features]
    y = ml_data[selected_target]

    # Handle missing values
    X = X.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(method='ffill').fillna(method='bfill')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model Training
    # Show progress bar with wastewater-themed steps
    progress_steps = [
        "Collecting samples from influent...",
        "Cleaning the bar screen...",
        "Checking sludge blankets and skimming...",
        "Topping off tanks...",
        "Adjusting chemistry and aeration...",
        "Optimizing blower speed and pump settings...",
        "Checking DO and effluent quality...",
        "Balancing flow rates between basins...",
        "Discharging treated water downstream...",
        "All systems go! Clean predictions discharged to the environment! 🎉"
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate progress
    for i, step in enumerate(progress_steps):
        progress = (i + 1) / len(progress_steps)
        status_text.text(step)
        progress_bar.progress(progress)
        time.sleep(0.5)  # Simulate time delay for effect

    # Reset progress bar
    progress_bar.empty()
    status_text.empty()

    # Train XGBoost model
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        objective='reg:squarederror',
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Trust Indicator Logic
    if r2 > 0.85 and rmse < y_test.std():
        trust_indicator = "🟢 High Confidence"
    elif 0.6 < r2 <= 0.85:
        trust_indicator = "🟡 Medium Confidence"
    else:
        trust_indicator = "🔴 Low Confidence"

    # Display Results
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("R² Score", f"{r2:.2f}")
    with col3:
        st.metric("Model Confidence", trust_indicator)

    # Feature Importances
    st.subheader("Feature Importances")
    importance_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in selected_features],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

    # Predictions vs Actual
    st.subheader("Predictions vs Actual")
    pred_df = pd.DataFrame({
        'Actual': y_test.reset_index(drop=True),
        'Predicted': y_pred,
        'Date': X_test.index
    }).sort_values(by='Date')
    fig = px.line(pred_df, x='Date', y=['Actual', 'Predicted'])
    st.plotly_chart(fig, use_container_width=True)

    # Download Option
    st.subheader("Download Results")
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )

    # Additional Option for Experts
    with st.expander("Optimization (Experimental)"):
        st.write("Optimize parameters to achieve desired target values.")
        st.write("This feature is under development.")
        st.info("Coming Soon!")
