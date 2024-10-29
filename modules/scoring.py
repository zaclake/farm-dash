# modules/scoring.py

import numpy as np
import pandas as pd
import streamlit as st

def calculate_feature_score(value, min_val, max_val, optimal_min, optimal_max):
    """Calculate individual feature scores using optimal ranges."""
    try:
        value = float(value)
        min_val = float(min_val)
        max_val = float(max_val)
        optimal_min = float(optimal_min)
        optimal_max = float(optimal_max)
    except (ValueError, TypeError):
        return np.nan

    if np.isnan([value, min_val, max_val, optimal_min, optimal_max]).any():
        return np.nan

    if min_val >= max_val or optimal_min >= optimal_max:
        return np.nan

    optimal_min = max(min_val, optimal_min)
    optimal_max = min(max_val, optimal_max)
    optimal_mid = (optimal_min + optimal_max) / 2

    if value < min_val or value > max_val:
        return 0
    elif optimal_min <= value <= optimal_max:
        if value <= optimal_mid:
            score = 1 - (optimal_mid - value) / (optimal_mid - optimal_min) * 0.1
        else:
            score = 1 - (value - optimal_mid) / (optimal_max - optimal_mid) * 0.1
        return np.clip(score, 0, 1)
    elif min_val <= value < optimal_min:
        score = 0.9 - ((optimal_min - value) / (optimal_min - min_val)) * 0.9
        return np.clip(score, 0, 0.9)
    elif optimal_max < value <= max_val:
        score = 0.9 - ((value - optimal_max) / (max_val - optimal_max)) * 0.9
        return np.clip(score, 0, 0.9)
    else:
        return 0

def calculate_difficulty_score(value, min_val, max_val, optimal_min, optimal_max):
    """Calculate difficulty feature scores (flipped)."""
    score = calculate_feature_score(value, min_val, max_val, optimal_min, optimal_max)
    if not np.isnan(score):
        return np.clip(1 - score, 0, 1)
    else:
        return np.nan

def compute_plant_scores(ml_data, configuration):
    """Compute current plant performance and difficulty scores."""
    # Implement the logic from your original script
    # This includes calculating unit process scores and overall plant score
    # Return a dictionary with all relevant scores
    # For brevity, this function needs to include all the steps from your original code
    # Ensure to handle exceptions and edge cases

    # Your original code logic here...
    # Due to space constraints, please integrate your existing code into this function

    # Example placeholder:
    plant_scores = {
        'plant_performance_score': 0.85,
        'difficulty_score': 0.30,
        'adjusted_performance_score': 0.88,
        'unit_process_scores': {},  # Fill with actual unit process scores
        'data_completeness': None,  # Fill with actual data completeness
    }

    return plant_scores

def calculate_scores_over_time(ml_data, configuration):
    """Calculate plant performance and difficulty scores over time."""
    # Implement the logic from your original script
    # Return a dictionary with time series data
    # For brevity, integrate your existing code into this function

    # Example placeholder:
    scores_over_time = {
        'dates': [],  # Fill with dates
        'overall_scores': [],  # Fill with overall scores
        'difficulty_scores': [],  # Fill with difficulty scores
        'adjusted_scores': [],  # Fill with adjusted scores
        'unit_process_scores': {},  # Fill with unit process scores over time
    }

    return scores_over_time
