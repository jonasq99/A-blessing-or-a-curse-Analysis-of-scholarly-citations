from src.information_extraction import (
    file_finder,
    load_annotations,
    format_author_name,
    df_to_triplets,
    dict_to_triplets,
    information_extraction,
    calculate_scores,
    evaluate_extraction,
    extract_citations,
    tagger_information_extraction,
)

# TODO: write unit tests for all functions except calculate_similarity (all imported functions)

import pandas as pd
import pytest


@pytest.mark.correct
def test_format_author_name():
    # Test format_author_name function
    result = format_author_name("John Doe")
    expected_result = "Doe, John"
    assert result == expected_result
    # Test case including and
    assert format_author_name("John Doe and Jane Doe") == "Doe, John and Doe, Jane"

@pytest.mark.correct
def test_df_to_triplets():
    # Test df_to_triplets function
    df = pd.DataFrame({"Footnote": [1, 2], "Authors": ["Author 1", "Author 2"], "Title": ["Title 1", "Title 2"]})
    result_triplets = df_to_triplets(df, format_author=False)
    expected_triplets = {(1, "Author 1", "Title 1"), (2, "Author 2", "Title 2")}
    assert result_triplets == expected_triplets

@pytest.mark.correct
def test_dict_to_triplets():
    # Test dict_to_triplets function
    extraction_dict = {1: [("Author 1", "Title 1")], 2: [("Author 2", "Title 2")]}
    result_triplets = dict_to_triplets(extraction_dict)
    expected_triplets = {(1, "Author 1", "Title 1"), (2, "Author 2", "Title 2")}
    assert result_triplets == expected_triplets
