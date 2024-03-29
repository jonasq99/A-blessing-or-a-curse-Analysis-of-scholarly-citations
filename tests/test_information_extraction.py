import json
import pandas as pd
import pytest
from src.information_extraction import (
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


@pytest.fixture
def sample_annotation_file(tmp_path):
    file_path = tmp_path / "sample_annotation.xlsx"
    df = pd.DataFrame(
        {
            "Footnote": ["1", "2", "3"],
            "Authors": ["Author 1", "nan", "Author 3"],
            "Title": ["Title 1", "Title 2", ""],
        }
    )
    df.to_excel(file_path, index=False)
    return str(file_path)


def test_load_annotations(sample_annotation_file):
    df = load_annotations(sample_annotation_file)
    expected_df = pd.DataFrame(
        {
            "Footnote": [1, 2, 3],
            "Authors": ["Author 1", None, "Author 3"],
            "Title": ["Title 1", "Title 2", None],
        }
    )
    assert len(df) == 3
    pd.testing.assert_frame_equal(df, expected_df)


def test_format_author_name():
    # Test format_author_name function
    result = format_author_name("John Doe")
    expected_result = "Doe, John"
    assert result == expected_result
    # Test case including and
    assert format_author_name("John Doe and Jane Doe") == "Doe, John and Doe, Jane"


def test_df_to_triplets():
    # Test df_to_triplets function
    df = pd.DataFrame(
        {
            "Footnote": [1, 2],
            "Authors": ["Author 1", "Author 2"],
            "Title": ["Title 1", "Title 2"],
        }
    )
    result_triplets = df_to_triplets(df, format_author=False)
    expected_triplets = {(1, "Author 1", "Title 1"), (2, "Author 2", "Title 2")}
    assert result_triplets == expected_triplets


def test_dict_to_triplets():
    # Test dict_to_triplets function
    extraction_dict = {1: [("Author 1", "Title 1")], 2: [("Author 2", "Title 2")]}
    result_triplets = dict_to_triplets(extraction_dict)
    expected_triplets = {(1, "Author 1", "Title 1"), (2, "Author 2", "Title 2")}
    assert result_triplets == expected_triplets


def test_calculate_scores():
    sample_triplets = {
        ("1", "Author 1", "Title 1"),
        ("1", "Author 2", "Title 2"),
        ("2", "Author 3", "Title 3"),
        ("3", "Author 4", "Title 4"),
    }
    sample_extraction = {
        ("1", "Author 1", "Title 1"),
        ("1", None, "Title 2"),
        ("2", "Author 3", None),
        ("3", "Author 4", "Title 4"),
    }
    recall, precision, f_score = calculate_scores(sample_triplets, sample_extraction)
    assert recall == 0.5
    assert precision == 0.5
    assert f_score == 0.5

    sample_triplets = {
        ("1", "Author 1", "Title 1"),
        ("1", "Author 2", "Title 2"),
        ("2", "Author 3", "Title 3"),
        ("3", "Author 4", "Title 4"),
    }
    sample_extraction = {("1", "Author 1", "Title 1"), ("2", "Author 3", None)}
    recall, precision, f_score = calculate_scores(sample_triplets, sample_extraction)
    assert recall == 0.25
    assert precision == 0.5
    assert f_score == 0.3333333333333333

    sample_triplets = {
        ("1", "Author 1", "Title 1"),
        ("2", "Author 3", "Title 3"),
        ("3", "Author 4", "Title 4"),
    }
    sample_extraction = {
        ("1", "Author 1", "Title 1"),
        ("1", None, "Title 2"),
        ("2", "Author 3", None),
        ("3", "Author 4", "Title 4"),
    }
    recall, precision, f_score = calculate_scores(sample_triplets, sample_extraction)
    assert recall == 0.6666666666666666
    assert precision == 0.5
    assert f_score == 0.5714285714285715


def test_evaluate_extraction():
    sample_triplets = {
        ("1", "Author 1", "Title 1"),
        ("1", "Author 2", "Title 2"),
        ("2", "Author 3", "Title 3"),
        ("3", "Author 4", "Title 4"),
    }
    sample_extraction = {
        ("1", "Author 1", "Title 1"),
        ("1", None, "Title 2"),
        ("2", "Author 3", None),
        ("3", "Author 4", "Title 4"),
    }
    precision, recall, f_score = evaluate_extraction(sample_triplets, sample_extraction)
    assert precision == 1.0
    assert recall == 1.0
    assert f_score == 1.0

    sample_triplets = {
        ("1", "Author 1", "Title 1"),
        ("1", "Author 2", "Title 2"),
        ("2", "Author 3", "Title 3"),
        ("3", "Author 4", "Title 4"),
    }
    sample_extraction = {("1", "Author 1", "Title 1"), ("2", "Author 3", None)}
    recall, precision, f_score = evaluate_extraction(sample_triplets, sample_extraction)
    assert recall == 1.0
    assert precision == 0.5
    assert f_score == 0.6666666666666666

    sample_triplets = {
        ("1", "Author 1", "Title 1"),
        ("2", "Author 3", "Title 3"),
        ("3", "Author 4", "Title 4"),
    }
    sample_extraction = {
        ("1", "Author 1", "Title 1"),
        ("1", None, "Title 2"),
        ("2", "Author 3", None),
        ("3", "Author 4", "Title 4"),
    }
    recall, precision, f_score = evaluate_extraction(sample_triplets, sample_extraction)
    assert recall == 0.75
    assert precision == 1.0
    assert f_score == 0.8571428571428571


def test_extract_citations(tmp_path):
    # Create a sample JSON file
    file_path = tmp_path / "sample.json"
    article = {
        "footnotes": {
            "1": "Author 1, Title 1 (location, 1999); Author 2, Title 2 (location, 1999).",
            "2": "Author 3, Title 3 (location, 1999).",
            "3": "Author 4, Title 4 (location, 1999).",
        }
    }
    with open(file_path, "w") as f:
        json.dump(article, f)

    # Test extract_citations function
    result_citations = extract_citations(str(file_path), path=str(tmp_path))
    expected_citations = {
        (1, "Author 1", "Title 1 "),
        (1, "Author 2", "Title 2 "),
        (2, "Author 3", "Title 3 "),
        (3, "Author 4", "Title 4 "),
    }
    assert result_citations == expected_citations

    # test if footnote is ibid
    article = {
        "footnotes": {
            "1": "Author 1, Title 1 (location, 1999).",
            "2": "Ibid",
            "3": "Author 2, Title 2 (location, 1999).",
        }
    }
    with open(file_path, "w") as f:
        json.dump(article, f)

    result_citations = extract_citations(str(file_path), path=str(tmp_path))
    expected_citations = {
        (1, "Author 1", "Title 1 "),
        (2, "Author 1", "Title 1 "),
        (3, "Author 2", "Title 2 "),
    }
    assert result_citations == expected_citations
