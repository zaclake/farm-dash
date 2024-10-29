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

    # Clean configuration column names: strip whitespace and convert to lowercase
    configuration.columns = configuration.columns.str.strip().str.lower()

    # Initialize session state variables
    if 'feature_importances_calculated' not in st.session_state:
        st.session_state['feature_importances_calculated'] = False
    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = []
    if 'learning_rate' not in st.session_state:
        st.session_state['learning_rate'] = 0.1
    if 'max_depth' not in st.session_state:
        st.session_state['max_depth'] = 6
    if 'n_estimators' not in st.session_state:
        st.session_state['n_estimators'] = 100

    # Step 1: Target Metric Selection
    target_options = [col for col in ml_data.columns if col != 'date']
    target_display_names = [col.replace('_', ' ').title() for col in target_options]
    target_name_mapping = dict(zip(target_display_names, target_options))
    selected_target_display = st.selectbox("Select Target Metric to Predict or Optimize", target_display_names)
    selected_target = target_name_mapping[selected_target_display]
    st.session_state['selected_target'] = selected_target

    # Add a submit button after target selection
    if st.button("Calculate Feature Importances"):
        # Set the flag to True
        st.session_state['feature_importances_calculated'] = True

    if st.session_state['feature_importances_calculated']:
        # Step 2: Initial Feature Importance Calculation
        st.subheader("Calculating Feature Importances...")
        with st.spinner('Calculating feature importances...'):
            feature_importances = calculate_feature_importance(ml_data, st.session_state['selected_target'])
        st.success("Feature importance calculation complete.")

        # Display Feature Importances
        st.subheader("Feature Importances")
        # Apply threshold to feature importances
        importance_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in feature_importances.index],
            'Importance': feature_importances.values
        })
        # Dynamic cutoff: remove features with importance less than 5% of the max importance
        max_importance = importance_df['Importance'].max()
        importance_df = importance_df[importance_df['Importance'] >= 0.05 * max_importance]
        # Sort and plot
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

        # Step 3: Feature Selection
        st.subheader("Select Features for the Model")
        # Filter features based on 'adjustability' column in configuration
        variable_features = get_variable_features(configuration)
        # Exclude the selected target from features
        available_features = [f for f in variable_features if f != st.session_state['selected_target']]
        if not available_features:
            st.warning("No variable features available for modeling.")
            return
        feature_display_names = [f.replace('_', ' ').title() for f in available_features]
        feature_name_mapping = dict(zip(feature_display_names, available_features))
        default_features = feature_display_names[:5] if not st.session_state['selected_features'] else [f.replace('_', ' ').title() for f in st.session_state['selected_features']]
        selected_feature_display = st.multiselect(
            "Select Features to Use in the Model (only variable features)",
            feature_display_names,
            default=default_features
        )
        selected_features = [feature_name_mapping[fdn] for fdn in selected_feature_display]
        st.session_state['selected_features'] = selected_features

        if not selected_features:
            st.warning("Please select at least one feature.")
            return

        # Step 4: Advanced Settings
        with st.expander("Advanced Settings"):
            st.write("Adjust hyperparameters for the XGBoost model.")
            st.session_state['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=st.session_state['learning_rate'], step=0.01)
            st.session_state['max_depth'] = st.slider("Max Depth", min_value=1, max_value=15, value=st.session_state['max_depth'], step=1)
            st.session_state['n_estimators'] = st.slider("Number of Estimators (Boosting Rounds)", min_value=50, max_value=500, value=st.session_state['n_estimators'], step=10)

        # Add a button to trigger model training and prediction
        if st.button("Run Model"):
            run_model(
                ml_data,
                st.session_state['selected_features'],
                st.session_state['selected_target'],
                st.session_state['learning_rate'],
                st.session_state['max_depth'],
                st.session_state['n_estimators']
            )

        # Optimization Section
        with st.expander("Optimization"):
            st.write("Optimize parameters to achieve desired target values.")

            # Allow users to input desired target value
            desired_target_value = st.number_input(
                f"Desired {selected_target_display} Value:",
                value=float(ml_data[st.session_state['selected_target']].mean())
            )
            st.session_state['desired_target_value'] = desired_target_value

            # Add a button to trigger optimization
            if st.button("Run Optimization"):
                run_optimization(
                    ml_data,
                    configuration,
                    st.session_state['selected_features'],
                    st.session_state['selected_target'],
                    st.session_state['learning_rate'],
                    st.session_state['max_depth'],
                    st.session_state['n_estimators'],
                    st.session_state['desired_target_value']
                )

    else:
        st.info("Please select a target metric and click 'Calculate Feature Importances'.")

def get_variable_features(configuration):
    # Clean 'adjustability' values
    if 'adjustability' not in configuration.columns:
        st.error("The 'adjustability' column is missing in the configuration data.")
        return []
    configuration['adjustability'] = configuration['adjustability'].astype(str).str.strip().str.lower()
    # Check if 'feature_name' column exists
    if 'feature_name' not in configuration.columns:
        st.error("The 'feature_name' column is missing in the configuration data.")
        return []
    # Proceed if columns exist
    variable_features = configuration[configuration['adjustability'] == 'variable']['feature_name'].tolist()
    return variable_features

def calculate_feature_importance(ml_data, selected_target):
    @st.cache_data
    def compute_importance(ml_data, selected_target):
        # Prepare data
        X = ml_data.drop(columns=['date', selected_target])
        y = ml_data[selected_target]

        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(method='ffill').fillna(method='bfill')

        # Convert data to DMatrix format
        dtrain = xgb.DMatrix(X, label=y)

        # Define parameters for XGBoost
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
        }

        # Train the model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            verbose_eval=False
        )

        # Get feature importances
        importance = model.get_score(importance_type='gain')
        importance_series = pd.Series(importance).sort_values(ascending=False)
        return importance_series

    importance_series = compute_importance(ml_data, selected_target)
    return importance_series

# The rest of the code remains unchanged (run_model, run_optimization, etc.)
# ...
