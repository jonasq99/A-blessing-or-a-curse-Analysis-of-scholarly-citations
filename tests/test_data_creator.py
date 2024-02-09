import pytest
from src.data_creator import footnote_to_int


def test_footnote_to_int():
    # Test with integer value
    assert footnote_to_int(5) == 5

    # Test with string representation of an integer
    assert footnote_to_int("10") == 10

    # Test with non-integer string value
    with pytest.raises(ValueError):
        footnote_to_int("footnote 10")

    # Test with None value
    with pytest.raises(TypeError):
        footnote_to_int(None)
