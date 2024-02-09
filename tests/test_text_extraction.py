import pytest
from src.text_extraction import TextExtraction


@pytest.mark.correct
def test_find_following_token_index():
    # Test find_following_token_index method
    sample_article_dict = {
        "article": "Lorem ipsum [CITATION-1] dolor sit amet, [CITATION-2] consectetur adipiscing elit.",
        "footnotes": {"1": "Sample footnote content.", "2": "Sample footnote content."},
    }
    text_extraction = TextExtraction(
        sample_article_dict,
        previous_context_tokens=None,
        following_context_tokens=4,
        footnote_text=False,
        footnote_mask=True,
    )

    # Test when following_context_tokens set to 4
    index = text_extraction.find_following_token_index(12)
    assert index == 41

    # Test when following_context_tokens is 0
    text_extraction.following_context_tokens = 0
    index = text_extraction.find_following_token_index(12)
    assert index == 12

    # Test when following_context_tokens is set to None
    text_extraction.following_context_tokens = None
    index = text_extraction.find_following_token_index(12)
    assert index == 12

    # Test when index reaches the end of the article_text
    text_extraction.following_context_tokens = 50
    index = text_extraction.find_following_token_index(12)
    assert index == len(sample_article_dict["article"])


@pytest.mark.correct
def test_find_previous_token_index():
    # Test find_previous_token_index method
    sample_article_dict = {
        "article": "Lorem ipsum [CITATION-1] dolor sit amet, [CITATION-2] consectetur adipiscing elit.",
        "footnotes": {"1": "Sample footnote content.", "2": "Sample footnote content."},
    }
    text_extraction = TextExtraction(
        sample_article_dict,
        previous_context_tokens=4,
        following_context_tokens=None,
        footnote_text=False,
        footnote_mask=True,
    )

    # Test when previous_context_tokens set to 4
    index = text_extraction.find_previous_token_index(41)
    assert index == 24

    # Test when previous_context_tokens is 0
    text_extraction.previous_context_tokens = 0
    index = text_extraction.find_previous_token_index(41)
    assert index == 41

    # Test when previous_context_tokens is set to None
    text_extraction.previous_context_tokens = None
    index = text_extraction.find_previous_token_index(41)
    assert index == 41

    # Test when index reaches the beginning of the article_text
    text_extraction.previous_context_tokens = 50
    index = text_extraction.find_previous_token_index(0)
    assert index == 0


@pytest.mark.correct
def test_replace_citations():
    # Test replacing sigle citation
    text = "Citation 1.[CITATION-1] Citation 2.[CITATION-2] This article"
    expected_replaced_text = "Citation 1.[CITATION-1] Citation 2.[MASK] This article"
    replaced_text = TextExtraction.replace_citations(text, 1)
    assert replaced_text == expected_replaced_text

    # Test replacing multiple citations
    text = "Citation 1.[CITATION-1] Citation 2.[CITATION-2] Citation 3.[CITATION-3] Citation 4.[CITATION-4] This article"
    expected_replaced_text = "Citation 1.[CITATION-1] Citation 2.[MASK] Citation 3.[MASK] Citation 4.[MASK] This article"
    replaced_text = TextExtraction.replace_citations(text, 1)
    assert replaced_text == expected_replaced_text

    # Test replacing citation with footnote number not in the text
    text = "Sample article text [CITATION-1]."
    expected_replaced_text = "Sample article text [MASK]."
    replaced_text = TextExtraction.replace_citations(text, 2)
    assert replaced_text == expected_replaced_text

    # Test replacing citations with no matches
    text = "Sample article text."
    expected_replaced_text = "Sample article text."
    replaced_text = TextExtraction.replace_citations(text, 1)
    assert replaced_text == expected_replaced_text


@pytest.mark.correct
def test_generate_context():
    sample_article_dict = {
        "article": "Lorem ipsum [CITATION-1] dolor sit amet, [CITATION-2] consectetur adipiscing elit.",
        "footnotes": {"1": "Sample footnote content.", "2": "Sample footnote content."},
    }

    text_extraction = TextExtraction(
        sample_article_dict,
        previous_context_tokens=3,
        following_context_tokens=None,
        footnote_text=False,
        footnote_mask=True,
    )

    # test with no following context
    generated_context = text_extraction.generate_context(2)
    expected_context = "sit amet, [CITATION-2]"
    assert generated_context == expected_context

    # test with no previous context
    text_extraction.previous_context_tokens = None
    text_extraction.following_context_tokens = 3
    generated_context = text_extraction.generate_context(2)
    expected_context = "[CITATION-2] consectetur adipiscing"
    assert generated_context == expected_context

    # test with both previous and following context
    text_extraction.previous_context_tokens = 4
    text_extraction.following_context_tokens = 3
    generated_context = text_extraction.generate_context(2)
    expected_context = "dolor sit amet, [CITATION-2] consectetur adipiscing"
    assert generated_context == expected_context

    # test with footnote text
    text_extraction.footnote_text = True
    generated_context = text_extraction.generate_context(2)
    expected_context = "dolor sit amet, [CITATION-2] consectetur adipiscing   \n   Footnote 2: Sample footnote content."
    assert generated_context == expected_context

    # test if footnote_mask=False works
    text_extraction.footnote_mask = False
    generated_context = text_extraction.generate_context(2)
    expected_context = "dolor sit amet, [CITATION-2] consectetur adipiscing   \n   Footnote 2: Sample footnote content."
    assert generated_context == expected_context
