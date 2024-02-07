## make test_get_context, test_few_shot_cot, test_get_fewshot_cot_examples, test_get_label


import sys
import os

from dotenv import load_dotenv

load_dotenv()
#apikey = os.getenv("OPENAI_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.utils import *

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

def test_get_completion(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "Mocked completion"
    
    result = get_completion("Test prompt")
    
    assert result == "Mocked completion"
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0,
    )

def test_get_completion_from_messages(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "Mocked completion from messages"

    result = get_completion_from_messages("Test messages")

    assert result == "Mocked completion from messages"
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages="Test messages",
        temperature=0,
        max_tokens=NOT_GIVEN, 
        top_p=NOT_GIVEN, 
        frequency_penalty=NOT_GIVEN, 
        presence_penalty=NOT_GIVEN
    )

def test_zero_shot(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "Mocked zero-shot response"

    result = zero_shot("Author Name", "Title", "Context", "Footnote")
    system_message = """
    You are an expert in analyzing citations from historical papers. It is your job 
    to determine if the author makes a statement about the quality of the work or just 
    uses it for the purpose of information reproduction.
    """
    prompt = f"""
    You will receive the name of the author of the cited source, its title, the context 
    of the citation and its corresponding footnote. 
    The data will be submitted in the following format:
    #######################Begin format instructions####################################
    name: name of authors
    title: title of cited source
    context: The context of the citation
    footnote: The corresponding footnote text of the citation
    #######################End format instructions#################################
    In the context citations are annotated like this: "[CITATION-footnotenumber]".

    The author makes a statement about the quality of the work if:
    - the author makes a judgemental statement about the quality of a  cited source. 
    - the author rates the quality of the work in a positive or negative mannser etc.
    Keywords of opinionated citations:
    - better, failed, argue, however, convincingy, nuanced, vague, fail, overlook, simplification, neglect

    A citation reproduces information if it does not make a statement about the quality of the cited work!

    Look closely at the text in the footnote! It can be the case that the hint if 
    a citation is neutral or opinionated might be located in the footnote text.
    Footnotes often contain multiple citations of different authors. 
    Look at the names of authors from the name field and relate them to the footnote. 
    Only rate the citation that is related to the name of authors in the name field. 

    Return 1 if the author makes a statement about the quality of the work else 0.
    Only return integers 0 and 1, nothing else.
    #################Begin data##################################
    Data:
    name: {"Author Name"}
    title: {"Title"}
    context: {"Context"}
    footnote: {"Footnote"}
    ###############End data#########################################
    Your answer: Enter integer here
    """

    assert result == "Mocked zero-shot response"
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        
        messages=[
            {"role": "system", "content": system_message}, ## erwartet die system message, die eine lokale variable ist
            {"role": "user", "content": prompt}, ## erwartet die prompt, die eine lokale Variable ist
        ],
        temperature=0,
        max_tokens=NOT_GIVEN, 
        top_p=NOT_GIVEN, 
        frequency_penalty=NOT_GIVEN, 
        presence_penalty=NOT_GIVEN
    )  

def test_calculate_accuracy_per_label():
    # Test case 1: No predictions and labels, expect 0 accuracy
    assert calculate_accuracy_per_label([], [], 1) == 0

    # Test case 2: No occurrences of the specified label, expect 0 accuracy
    assert calculate_accuracy_per_label([0, 0, 0], [1, 1, 1], 0) == 0

    # Test case 3: Some occurrences of the specified label, but no correct predictions, expect 0 accuracy
    assert calculate_accuracy_per_label([0, 0, 0], [0, 1, 0], 1) == 0

    # Test case 4: Some occurrences of the specified label, with correct predictions, expect non-zero accuracy
    assert calculate_accuracy_per_label([1, 1, 1, 0, 0], [1, 1, 1, 0, 0], 1) == 1.0

    # Test case 5: Some occurrences of the specified label, with some correct predictions, expect non-zero accuracy
    assert calculate_accuracy_per_label([0, 0, 1, 1, 0], [0, 1, 0, 1, 0], 1) == 0.5
