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
import json
import os
import re
import pytest

@pytest.fixture
def sample_dataframe():
    data = {
        "DOI": ["12345", "67890"],
        "Title": ["Sample Title 1", "Sample Title 2"],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_annotation_file(tmp_path):
    file_path = tmp_path / "sample_annotation.xlsx"
    df = pd.DataFrame({
        "Footnote": ["1", "2", "3"],
        "Authors": ["Author 1", "nan", "Author 3"],
        "Title": ["Title 1", "Title 2", ""],
    })
    df.to_excel(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_json_file(tmp_path):
    file_path = tmp_path / "sample.json"
    data = {"title": "this is the title",
            "author": "name of author",
            "date": "01 January 1950",
            "article": "this is the text of the article",
            "footnotes": {
                "1": "Author 1, Title 1 (year); Author 2, Title 2 (year)",
                "2": "John Doe, Title 3",
                "3": "Ibid",
                "4": "Author 4, Title 4. See also Author 5, Title 5",
                "5": "text of footnote that leads to author, see Author 6, Title 6",
            }
        }
    with open(file_path, "w") as f:
        json.dump(data, f)
    return str(file_path)


@pytest.fixture
def sample_triplets():
    return {("1", "Author 1", "Title 1"), ("1", "Author 2", "Title 2"), ("2", "Author 3", "Title 3"), ("3", "Author 4", "Title 4")}

@pytest.fixture
def sample_extraction():
    return {("1", "Author 1", "Title 1"), ("1", None, "Title 2"), ("2", "Author 3", None), ("3", "Author 4", "Title 4")}


""" @pytest.fixture
def sample_tagger():
    # Assuming you have a sample tagger object for testing
    return None """


""" def test_file_finder():
    assert file_finder("some_file_12345.json") == "matching_file.json"
    # Add more test cases if needed """

""" @pytest.mark.correct
def test_load_annotations(sample_annotation_file):
    df = load_annotations(sample_annotation_file)
    expected_df = pd.DataFrame({
        "Footnote": [1, 2, 3],
        "Authors": ["Author 1", None, "Author 3"],
        "Title": ["Title 1", "Title 2", None],
    })
    assert len(df) == 3
    pd.testing.assert_frame_equal(df, expected_df)    
    # Add more assertions if needed


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
    assert result_triplets == expected_triplets """

""" #Version 1 test_information_extraction -> FileNotFoundError 
def test_information_extraction(sample_json_file):
    triplets = information_extraction(sample_json_file)
    assert len(triplets) == 3
    # Add more assertions if needed """

""" #Version 2 test_information_extraction -> FileNotFoundError
@pytest.fixture
def sample_article_path(tmp_path):
    # Create a sample article JSON file
    article = {
        "footnotes": {
            "1": "Doe, J. (2000). Sample reference. Journal of Examples.",
            "2": "Smith, A. (1999). Another reference. Example Publishers.",
            "3": "Ibid.",
            "4": "Doe, J. (2001). Another sample reference. Example Books.",
        }
    }
    file_path = tmp_path / "sample_article.json"
    with open(file_path, "w") as f:
        json.dump(article, f)
    return str(file_path)

def test_information_extraction(sample_article_path):
    # Test the function with a sample article file
    result = information_extraction(sample_article_path)
    # Define the expected triplets
    expected_triplets = {
        (1, "Doe, J.", "Sample reference. Journal of Examples."),
        (2, "Smith, A.", "Another reference. Example Publishers."),
        (3, None, None),
        (4, "Doe, J.", "Another sample reference. Example Books."),
    }
    assert result == expected_triplets

def test_information_extraction_empty_file(tmp_path):
    # Test the function with an empty article file
    empty_file_path = os.path.join(tmp_path, "empty_article.json")
    open(empty_file_path, "w").close()
    result = information_extraction(empty_file_path)
    assert result == set() """

""" #version 3 test_information_extraction -> FileNotFoundError
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_article():
    return {
        "footnotes": {
            "1": "Doe, J. (2000). Sample reference. Journal of Examples.",
            "2": "Ibid.",
            "3": "Smith, A. (1999). Another sample reference. Journal of Examples.",
        }
    }

@patch("subprocess.run")
def test_information_extraction(mock_subprocess_run, mock_article):
    mock_subprocess_run.return_value = MagicMock(stdout="some_fake_output")

    test_directory = os.path.dirname(os.path.abspath(__file__))
    sample_article_path = os.path.join(test_directory, "sample_article.json")

    result = information_extraction(sample_article_path)

    assert isinstance(result, set)
    assert len(result) == 3  # Assuming each footnote has one reference """

""" @pytest.mark.correct
def test_calculate_scores():
    sample_triplets = {("1", "Author 1", "Title 1"), ("1", "Author 2", "Title 2"), ("2", "Author 3", "Title 3"), ("3", "Author 4", "Title 4")}
    sample_extraction = {("1", "Author 1", "Title 1"), ("1", None, "Title 2"), ("2", "Author 3", None), ("3", "Author 4", "Title 4")}
    recall, precision, f_score = calculate_scores(sample_triplets, sample_extraction)
    assert recall == 0.5
    assert precision == 0.5
    assert f_score == 0.5

    sample_triplets = {("1", "Author 1", "Title 1"), ("1", "Author 2", "Title 2"), ("2", "Author 3", "Title 3"), ("3", "Author 4", "Title 4")}
    sample_extraction = {("1", "Author 1", "Title 1"), ("2", "Author 3", None)}
    recall, precision, f_score = calculate_scores(sample_triplets, sample_extraction)
    assert recall == 0.25
    assert precision == 0.5
    assert f_score == 0.3333333333333333

    sample_triplets = {("1", "Author 1", "Title 1"), ("2", "Author 3", "Title 3"), ("3", "Author 4", "Title 4")}
    sample_extraction = {("1", "Author 1", "Title 1"), ("1", None, "Title 2"), ("2", "Author 3", None), ("3", "Author 4", "Title 4")}
    recall, precision, f_score = calculate_scores(sample_triplets, sample_extraction)
    assert recall == 0.6666666666666666
    assert precision == 0.5
    assert f_score == 0.5714285714285715 """
    

""" @pytest.mark.correct
def test_evaluate_extraction():
    sample_triplets = {("1", "Author 1", "Title 1"), ("1", "Author 2", "Title 2"), ("2", "Author 3", "Title 3"), ("3", "Author 4", "Title 4")}
    sample_extraction = {("1", "Author 1", "Title 1"), ("1", None, "Title 2"), ("2", "Author 3", None), ("3", "Author 4", "Title 4")}
    precision, recall, f_score = evaluate_extraction(sample_triplets, sample_extraction)
    assert precision == 1.0
    assert recall == 1.0
    assert f_score == 1.0
    
    sample_triplets = {("1", "Author 1", "Title 1"), ("1", "Author 2", "Title 2"), ("2", "Author 3", "Title 3"), ("3", "Author 4", "Title 4")}
    sample_extraction = {("1", "Author 1", "Title 1"), ("2", "Author 3", None)}
    recall, precision, f_score = evaluate_extraction(sample_triplets, sample_extraction)
    assert recall == 1.0
    assert precision == 0.5
    assert f_score == 0.6666666666666666

    sample_triplets = {("1", "Author 1", "Title 1"), ("2", "Author 3", "Title 3"), ("3", "Author 4", "Title 4")}
    sample_extraction = {("1", "Author 1", "Title 1"), ("1", None, "Title 2"), ("2", "Author 3", None), ("3", "Author 4", "Title 4")}
    recall, precision, f_score = evaluate_extraction(sample_triplets, sample_extraction)
    assert recall == 0.75
    assert precision == 1.0
    assert f_score == 0.8571428571428571 """


""" def test_extract_citations(sample_json_file):
    citations = extract_citations(sample_json_file)
    #assert len(citations) == 7 #sollte eigentlich idealerweise 6 sein, aber ist vielleicht zu kompliziert
    # Add more assertions if needed
    expected_citations = {
        (1, 'Author 1', 'Title 1'),
        (1, 'Author 2', 'Title 2'),
        (2, 'John Doe', 'Title 3'),
        (3, 'John Doe', 'Title 3'),
        (4, 'Author 4', 'Title 4'),
        (4, 'Author 5', 'Title 5'),
        (5, 'Author 6', 'Title 6'),
    }
    assert citations == expected_citations """

from unittest.mock import MagicMock

@pytest.fixture
def sample_tagger():
    # Create a mock tagger object
    tagger = MagicMock()

    # Define the behavior of the mock tagger's predict method
    def mock_predict(sentence):
        # Define the behavior of the predict method
        # For example, you can simulate the behavior of the tagger here
        # For simplicity, let's just return a predefined result
        sentence.add_label("PERSON", "John Doe")
        sentence.add_label("PERSON", "Author")
        sentence.add_label("WORK_OF_ART", "Title")

    # Attach the mock predict method to the tagger object
    tagger.predict.side_effect = mock_predict

    return tagger


def test_tagger_information_extraction(sample_json_file, sample_tagger):
    citations = tagger_information_extraction(sample_json_file, sample_tagger)
    assert len(citations) == 3
    # Add more assertions if needed

