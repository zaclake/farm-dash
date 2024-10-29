# dashboard.py

import streamlit as st
from modules import dropbox_integration
from modules import data_preprocessing
from modules import configuration_handler
from modules import scoring
from modules import visualization

def run_dashboard():
    st.set_page_config(page_title="Wastewater Treatment Plant Dashboard", layout="wide")
    st.title("Wastewater Treatment Plant Dashboard")

    # Initialize Dropbox and download data
    dbx = dropbox_integration.initialize_dropbox()
    if not dbx:
        return

    data_downloaded = dropbox_integration.download_data_file(dbx)
    if not data_downloaded:
        return

    # Load data
    ml_data, configuration = data_preprocessing.load_data("daily_data.xlsx")
    if ml_data is None or configuration is None:
        return

    # Preprocess data
    ml_data = data_preprocessing.preprocess_ml_data(ml_data)
    if ml_data is None:
        return

    configuration = configuration_handler.preprocess_configuration(configuration)
    if configuration is None:
        return

    # Validate data
    valid = data_preprocessing.validate_data(ml_data, configuration)
    if not valid:
        return

    # Compute scores
    plant_scores = scoring.compute_plant_scores(ml_data, configuration)
    if plant_scores is None:
        return

    # Extract variables for visualization
    unit_process_scores = plant_scores['unit_process_scores']
    formatted_unit_process_names = plant_scores['formatted_unit_process_names']
    data_completeness = plant_scores['data_completeness']

    # Calculate scores over time
    scores_over_time = scoring.calculate_scores_over_time(ml_data, configuration)

    # Visualize results
    visualization.run_visualization(
        plant_scores,
        scores_over_time,
        unit_process_scores,
        formatted_unit_process_names,
        data_completeness,
        ml_data
    )

if __name__ == "__main__":
    run_dashboard()
