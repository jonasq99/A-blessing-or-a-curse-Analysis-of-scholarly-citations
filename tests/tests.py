import sys
from pathlib import Path
import pytest

from src.utils import get_context
from src.data_creator import create_data


@pytest.mark.correct
def test_get_context():
    # Test case 1: No [MASK], no succeeding context
    text = "Wilhelm II's condemnation of Cecil Rhodes as a 'monstrous villain' in response to the Jameson Raid into Transvaal [CITATION-2] reveals his strong disapproval of Rhodes' actions. This suggests that Wilhelm II held a negative opinion towards Rhodes and his involvement in the scandal."
    expected_context = "Wilhelm II's condemnation of Cecil Rhodes as a 'monstrous villain' in response to the Jameson Raid into Transvaal [CITATION-2]"
    assert get_context(text) == expected_context
    # Test case 2: No [MASK], with succeeding context
    text = (
        "The end of the nineteenth century marked a significant shift in politics, where political figures had to adapt to the changing landscape of media attention. This shift not only paved the way for the rise of figures like Cecil Rhodes but also laid the foundation for the subsequent criticism of his cult and its association with racism in recent times. As the Times reported, "
        "Among the most picturesque incidents of an age of intercommunication must be reckoned the visit of Mr. Rhodes to Berlin"
        " [CITATION-1]."
    )
    expected_context = (
        "The end of the nineteenth century marked a significant shift in politics, where political figures had to adapt to the changing landscape of media attention. This shift not only paved the way for the rise of figures like Cecil Rhodes but also laid the foundation for the subsequent criticism of his cult and its association with racism in recent times. As the Times reported, "
        "Among the most picturesque incidents of an age of intercommunication must be reckoned the visit of Mr. Rhodes to Berlin"
        " [CITATION-1]."
    )
    assert get_context(text) == expected_context
