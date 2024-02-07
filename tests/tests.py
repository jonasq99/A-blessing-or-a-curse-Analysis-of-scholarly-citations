from src.utils import get_context
from src.data_creator import create_data
from src.synthetic_data_generation import create_opinionated


# import pytest


def test_get_context():
    # Test case 1: No [MASK] or CITATION tokens, expect the same text as context
    text = "This is a sample text."
    expected_context = "This is a sample text."
    assert get_context(text) == expected_context

    # Test case 2: [MASK] token found, expect an empty context
    text = "This is a [MASK] text."
    expected_context = ""
    assert get_context(text) == expected_context

    # Test case 3: CITATION token found, expect the context up to the CITATION token
    text = "This is a sample CITATION text."
    expected_context = "This is a sample"
    assert get_context(text) == expected_context

    # Test case 4: Multiple [MASK] tokens, expect an empty context
    text = "[MASK] [MASK] [MASK]"
    expected_context = ""
    assert get_context(text) == expected_context

    # Test case 5: [MASK] token followed by CITATION token, expect an empty context
    text = "[MASK] CITATION"
    expected_context = ""
    assert get_context(text) == expected_context
