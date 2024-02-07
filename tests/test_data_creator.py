# revise all
# hat nichts geändert, dass jetzt die create_data function in einem einzelnen file ist
# es wird trotzdem nicht über die funktion die richtigen weiteren funktionen importiert

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.text_extraction import file_finder, TextExtraction
from src.data_creator import create_data



""" import sys
import os

from dotenv import load_dotenv

load_dotenv()
#apikey = os.getenv("OPENAI_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.utils import * """

import pytest
import pandas as pd
from pathlib import Path
import json
from unittest.mock import Mock, patch

@pytest.fixture
def mock_openai_client():
    with patch("src.utils.client") as mock_client:
        yield mock_client

@pytest.fixture
def example_dataframes():
    # Create some example DataFrames for testing
    df1 = pd.DataFrame({'Label': [1, 2, 3], 'Value': [10, 20, 30]})
    df2 = pd.DataFrame({'Label': [1, 2, 3], 'Value': [15, 25, 35]})
    df3 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Value': [5, 10, 15]})
    return {'df1': df1, 'df2': df2, 'df3': df3}


def test_create_data():
    # Mock file_finder and TextExtraction functions for this test
    with patch("file_finder", return_value="mocked_title.json") as mock_file_finder, \
         patch("TextExtraction") as mock_text_extraction:
        
        """ # Change directory to the same of the function file_finder?
        src_dir = os.path.join(os.getcwd(),"src")
        os.chdir(src_dir)
        print(os.getcwd())
 """
        # Mock the return value of generate_context in TextExtraction
        mock_text_extraction.return_value.generate_context.return_value = "mocked_context"

        result = create_data(previous_context_tokens=5, following_context_tokens=5) ## pfad in utils 61/62 muss relativ gemacht werden

        assert len(result) == 1
        assert "mocked_df_name" in result
        df = result["mocked_df_name"]
        assert "context" in df.columns
        assert "footnote_text" in df.columns
        assert df["context"].tolist() == ["mocked_context"] * len(df)
        assert df["footnote_text"].tolist() == ["mocked_footnote"] * len(df)

        # Ensure the expected calls to file_finder and TextExtraction were made
        mock_file_finder.assert_called_once_with("mocked_df_name")
        mock_text_extraction.assert_called_once_with(
            {"footnotes": {"1": "mocked_footnote"}},
            previous_context_tokens=5,
            following_context_tokens=5,
            previous_context_sentences=None,
            following_context_sentences=None,
            previous_whole_paragraph=False,
            following_whole_paragraph=False,
            till_previous_citation=None,
            till_following_citation=None,
            footnote_text=False,
            footnote_mask=True,
        ) 
