# modules/machine_learning.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import time
from scipy.optimize import minimize

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
        n_estimators = st.slider("Number of Estimators (Boosting Rounds)", min_value=50, max_value=500, value=100, step=10)

    # Add a button to trigger model training and prediction
    if st.button("Run Model"):
        run_model(ml_data, selected_features, selected_target, learning_rate, max_depth, n_estimators)

    # Optimization Section
    with st.expander("Optimization"):
        st.write("Optimize parameters to achieve desired target values.")

        # Allow users to input desired target value
        desired_target_value = st.number_input(f"Desired {selected_target_display} Value:", value=float(ml_data[selected_target].mean()))

        # Add a button to trigger optimization
        if st.button("Run Optimization"):
            run_optimization(ml_data, configuration, selected_features, selected_target, learning_rate, max_depth, n_estimators, desired_target_value)

def run_model(ml_data, selected_features, selected_target, learning_rate, max_depth, n_estimators):
    st.subheader("Model Training and Prediction")

    # Prepare data
    X = ml_data[selected_features]
    y = ml_data[selected_target]

    # Handle missing values
    X = X.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(method='ffill').fillna(method='bfill')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Custom training loop
    eval_set = [(X_test, y_test)]
    eval_metric = ['rmse']
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=1,  # We will increment this in the loop
        objective='reg:squarederror',
        verbosity=0,
        warm_start=True  # Allows incremental training
    )

    for i in range(n_estimators):
        model.n_estimators = i + 1  # Increment the number of estimators
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=eval_metric,
            xgb_model=model if i > 0 else None,
            verbose=False
        )
        progress = (i + 1) / n_estimators
        progress_bar.progress(progress)
        status_text.text(f"Training Iteration: {i + 1}/{n_estimators}")

    # Clear progress bar
    progress_bar.empty()
    status_text.empty()

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

def run_optimization(ml_data, configuration, selected_features, selected_target, learning_rate, max_depth, n_estimators, desired_target_value):
    st.subheader("Parameter Optimization")

    # Prepare data
    X = ml_data[selected_features]
    y = ml_data[selected_target]

    # Handle missing values
    X = X.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(method='ffill').fillna(method='bfill')

    # Train the model on all data
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        objective='reg:squarederror',
        verbosity=0
    )
    model.fit(X, y)

    # Optimization logic
    # Define the objective function for optimization
    def objective_function(feature_values):
        # Convert feature_values to DataFrame
        feature_dict = {feature: [value] for feature, value in zip(selected_features, feature_values)}
        input_df = pd.DataFrame(feature_dict)

        # Predict using the model
        predicted = model.predict(input_df)[0]

        # Objective is the squared difference between predicted and desired target value
        return (predicted - desired_target_value) ** 2

    # Get bounds for features from configuration
    config = configuration.set_index('feature_name')
    bounds = []
    for feature in selected_features:
        if feature in config.index:
            min_val = config.loc[feature, 'min']
            max_val = config.loc[feature, 'max']
            bounds.append((min_val, max_val))
        else:
            # If feature not in configuration, use data min and max
            min_val = ml_data[feature].min()
            max_val = ml_data[feature].max()
            bounds.append((min_val, max_val))

    # Initial guess: mean of the feature values
    initial_guess = [ml_data[feature].mean() for feature in selected_features]

    # Maximum number of iterations
    max_iterations = 100

    # Run optimization
    st.info("Running optimization...")

    # Show progress bar during optimization
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Since we cannot use callback, we'll simulate progress updates
    iteration = [0]  # Mutable object to keep track of iterations

    def callback(xk):
        iteration[0] += 1
        progress = min(iteration[0] / max_iterations, 1.0)
        status_text.text(f"Optimization Iteration: {iteration[0]}")
        progress_bar.progress(progress)

    result = minimize(
        objective_function,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': max_iterations}
    )

    # Clear progress bar
    progress_bar.empty()
    status_text.empty()

    if result.success:
        st.success("Optimization converged successfully!")
    else:
        st.warning("Optimization did not converge.")

    optimized_feature_values = result.x
    optimized_features = {feature: value for feature, value in zip(selected_features, optimized_feature_values)}

    # Display optimized parameters
    st.subheader("Optimized Parameters")
    optimized_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in selected_features],
        'Optimized Value': optimized_feature_values
    })
    st.table(optimized_df)

    # Predicted value with optimized parameters
    optimized_prediction = model.predict(pd.DataFrame([optimized_feature_values], columns=selected_features))[0]
    st.subheader("Optimized Prediction")
    st.write(f"Predicted {selected_target.replace('_', ' ').title()}: {optimized_prediction:.2f}")

    # Trust Indicator Logic for Optimization
    if result.success and iteration[0] < (0.8 * max_iterations):
        trust_indicator = "🟢 High Confidence"
    elif iteration[0] < max_iterations:
        trust_indicator = "🟡 Medium Confidence"
    else:
        trust_indicator = "🔴 Low Confidence"

    st.metric("Optimization Confidence", trust_indicator)

    # Download Option for Optimization Results
    st.subheader("Download Optimization Results")
    optimization_results = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in selected_features],
        'Optimized Value': optimized_feature_values
    })
    optimization_results['Desired Target'] = desired_target_value
    optimization_results['Predicted Target'] = optimized_prediction

    csv = optimization_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Optimization Results as CSV",
        data=csv,
        file_name='optimization_results.csv',
        mime='text/csv'
    )
