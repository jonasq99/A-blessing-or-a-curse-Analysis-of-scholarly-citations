## problem of importing utils through synthetic_data_generation

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.synthetic_data_generation import *

import pytest
from unittest.mock import patch

@pytest.fixture
def mock_get_completion():
    # Mock the get_completion_from_messages function
    with patch("your_module.get_completion_from_messages") as mock:
        yield mock

def test_create_opinionated(mock_get_completion):
    prompt = "This is a sample prompt."
    
    # Replace this with the expected result for your test case
    expected_result = "This is an opinionated completion."
    
    # Configure the mock return value for get_completion_from_messages
    mock_get_completion.return_value = expected_result
    
    result = create_opinionated(prompt)
    
    assert result == expected_result
    mock_get_completion.assert_called_once_with(
        # Replace this with the expected messages for your test case
        [{"role": "system", "content": "system_message"}, {"role": "user", "content": prompt}],
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )