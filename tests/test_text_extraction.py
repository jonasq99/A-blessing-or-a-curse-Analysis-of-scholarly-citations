import pytest
from src.text_extraction import TextExtraction

# TODO: Write unit tests for the file_finder and TextExtraction


@pytest.mark.correct
def test_text_extraction():
    # Test TextExtraction class
    sample_article_dict = {
        "article": "Sample article text [CITATION-1].",
        "footnotes": {"1": "Sample footnote content."},
    }
    text_extraction = TextExtraction(sample_article_dict)

    # Test generate_context method
    context = text_extraction.generate_context(1)
    expected_context = "[CITATION-1]   \n   Footnote 1: Sample footnote content."
    assert context == expected_context
