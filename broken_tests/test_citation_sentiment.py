## problems with module/ importing utils

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.citation_sentiment import *

import pandas as pd
import pytest


""" @pytest.fixture
def example_data():
    # Replace this with appropriate data for your tests
    df_dict = create_data(70, 30)
    opinionated_data = filter_label(df_dict, 1)
    neutral_data = sample_data(filter_label(df_dict, 0))
    df = pd.concat([opinionated_data, neutral_data], ignore_index=True)
    return df """

@pytest.fixture
def example_dataframes_dict():
    # Replace this with appropriate data for your tests
    df1 = pd.DataFrame({'Label': [1, 0, 1], 'Data': ['A', 'B', 'C']})
    df2 = pd.DataFrame({'Label': [0, 1, 0], 'Data': ['X', 'Y', 'Z']})
    dataframes_dict = {'df1': df1, 'df2': df2}
    return dataframes_dict

def test_filter_label(example_data):
    label_1_df = filter_label(example_data, 1)
    assert all(label_1_df['Label'] == 1)

def test_sample_data(example_data):
    sampled_df = sample_data(example_data)
    assert len(sampled_df) == 100

def test_get_predictions(example_data, mocker):
    # Mocking the zero_shot function
    mocker.patch('script.zero_shot', return_value='1')

    predictions = get_precictions(example_data)
    assert all(p == 1 for p in predictions)

def test_calculate_accuracy_per_label():
    # Assuming you have predictions and labels for the test
    predictions = [1, 0, 1, 1, 0]
    labels = [1, 1, 0, 1, 0]

    accuracy_label_0 = calculate_accuracy_per_label(predictions, labels, label_value=0)
    accuracy_label_1 = calculate_accuracy_per_label(predictions, labels, label_value=1)

    assert accuracy_label_0 == 0.5  # Update with the expected accuracy
    assert accuracy_label_1 == 0.75  # Update with the expected accuracy