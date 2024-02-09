## pfad in information_extraction kann nicht gefunden werden (wegen ungleichem Pfadausgangspunkt)

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.information_extraction import *

import pandas as pd
import pytest

@pytest.fixture
def sample_title_doi_csv(tmp_path):
    # Create a temporary titles_doi.csv file
    csv_content = "DOI,Title\n123,Sample_Title"
    csv_path = tmp_path / "titles_doi.csv"
    csv_path.write_text(csv_content)
    return csv_path

""" @pytest.fixture
def sample_annotated_excel(tmp_path):
    # Create a temporary annotated Excel file
    excel_content = "Footnote,Authors,Title\n1,Author 1,Title 1\n2,Author 2,Title 2"
    excel_path = tmp_path / "sample_annotations.xlsx"
    pd.DataFrame({"Footnote": [1, 2], "Authors": ["Author 1", "Author 2"], "Title": ["Title 1", "Title 2"]}).to_excel(excel_path, index=False)
    return excel_path

def test_file_finder(sample_title_doi_csv, tmp_path):
    # Create a temporary JSON file with the expected title
    json_filename = "Sample_Title.json"
    json_path = tmp_path / json_filename
    json_path.write_text("")

    # Test file_finder function
    result = file_finder(f"file_123_{json_filename}")
    assert result == json_filename

def test_load_annotations(sample_annotated_excel):
    # Test load_annotations function
    result_df = load_annotations("sample_annotations.xlsx")
    expected_df = pd.DataFrame({"Footnote": [1, 2], "Authors": ["Author 1", "Author 2"], "Title": ["Title 1", "Title 2"]})
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_format_author_name():
    # Test format_author_name function
    result = format_author_name("John Doe")
    expected_result = "Doe, John"
    assert result == expected_result

def test_df_to_triplets():
    # Test df_to_triplets function
    df = pd.DataFrame({"Footnote": [1, 2], "Authors": ["Author 1", "Author 2"], "Title": ["Title 1", "Title 2"]})
    result_triplets = df_to_triplets(df, format_author=False)
    expected_triplets = {(1, "Author 1", "Title 1"), (2, "Author 2", "Title 2")}
    assert result_triplets == expected_triplets

def test_dict_to_triplets():
    # Test dict_to_triplets function
    extraction_dict = {1: [("Author 1", "Title 1")], 2: [("Author 2", "Title 2")]}
    result_triplets = dict_to_triplets(extraction_dict)
    expected_triplets = {(1, "Author 1", "Title 1"), (2, "Author 2", "Title 2")}
    assert result_triplets == expected_triplets """


#Version two
"""     import os
import pandas as pd
import pytest
from your_module import file_finder, load_annotations """
from unittest.mock import patch

@pytest.fixture
def mock_title_doi(monkeypatch):
    # Mock the content of the titles_doi.csv file
    titles_doi_content = """
    DOI,Title
    1234-5678,Sample_Title_1234
    5678-9012,Another_Title_5678)
    """
    monkeypatch.setenv("TITLES_DOI_CONTENT", titles_doi_content)

def test_file_finder(mock_title_doi):
    # Replace this with the actual file you want to test
    file_str = "sample_file_1234-5678.txt"
    
    # Replace this with the expected result for your test case
    expected_result = "Sample_Title_1234.json"
    
    result = file_finder(file_str)
    
    assert result == expected_result

def test_load_annotations(tmpdir):
    # Create a temporary directory for testing
    folder_path = tmpdir.mkdir("annotated")
    
    # Create a sample annotation file
    file_content = """
    Footnote,Authors,Title
    1,Author1,Title1
    2,Author2,Title2
    """
    file_path = folder_path.join("sample_annotation.xlsx")
    file_path.write(file_content)
    
    # Replace this with the expected DataFrame for your test case
    expected_df = pd.DataFrame({
        "Footnote": [1, 2],
        "Authors": ["Author1", "Author2"],
        "Title": ["Title1", "Title2"]
    })
    
    with patch("your_module.path_annotations", str(folder_path)):
        result_df = load_annotations("sample_annotation.xlsx")
    
    pd.testing.assert_frame_equal(result_df, expected_df)