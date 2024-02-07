## problem with importing utils trough impact_cite_testrun

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from src.impact_cite_testrun import *
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sre_constants


@pytest.fixture
def mock_create_data():
    # Mock the create_data function
    with patch("your_module.create_data") as mock_create_data:
        # Replace this with appropriate data for your tests
        mock_create_data.return_value = {
            "dataset_1": pd.DataFrame({
                "Label": [0, 1],
                "context": ["context1", "context2"],
                "footnote_text": ["footnote1", "footnote2"]
            }),
            "dataset_2": pd.DataFrame({
                "Label": [1, 0],
                "context": ["context3", "context4"],
                "footnote_text": ["footnote3", "footnote4"]
            })
            # Add more datasets as needed
        }
        yield mock_create_data


@pytest.fixture
def mock_model_tokenizer():
    # Mock the AutoModelForSequenceClassification and AutoTokenizer classes
    with patch("your_module.AutoModelForSequenceClassification") as mock_model, \
         patch("your_module.AutoTokenizer") as mock_tokenizer:
        # Replace these with appropriate mock behavior for your tests
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        yield mock_model, mock_tokenizer


def test_run(mock_create_data, mock_model_tokenizer):
    preceeding_context = 300
    suceeding_context = 100

    # Replace these with appropriate mock behavior for your tests
    mock_model, mock_tokenizer = mock_model_tokenizer

    with patch("your_module.calculate_accuracy_per_label") as mock_accuracy:
        mock_accuracy.return_value = 0.75  # Replace with your expected accuracy value

        result = run(preceeding_context, suceeding_context)

        assert "f1" in result
        assert "accuracy_0" in result
        assert "accuracy_1" in result
        # Replace these assertions with your expected values
        assert result["f1"] == 0.8
        assert result["accuracy_0"] == 0.75
        assert result["accuracy_1"] == 0.75


def test_run_configs(mock_create_data, mock_model_tokenizer):
    # Replace these with appropriate mock behavior for your tests
    mock_model, mock_tokenizer = mock_model_tokenizer

    # Replace this with appropriate data for your tests
    configs_testrun = [(300, 100)]

    with patch("your_module.open", create=True):
        run_configs(configs_testrun, file_path="test_results.csv")

    # Perform assertions based on the expected interactions with mocks
    # This will depend on how you expect the function to interact with the mocks