## revise test_file_finder (IndexError)

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.text_extraction import *

import pandas as pd
import pytest

@pytest.fixture
def sample_article_dict():
    return {
        "article": "Sample article text [CITATION-1].",
        "footnotes": {"1": "Sample footnote content."},
    }

def test_file_finder(tmpdir):

    # Change directory to the same of the fuction file_finder
    src_dir = os.path.join(os.getcwd(),"src")
    os.chdir(src_dir)

    # Create a temporary titles_doi.csv file
    titles_doi_content = "DOI,Title\n123,Sample_Title"
    titles_doi_path = tmpdir.join("titles_doi.csv")
    titles_doi_path.write(titles_doi_content)

    # Create a temporary all_data_articles directory with a sample JSON file
    all_data_articles_path = tmpdir.mkdir("all_data_articles")
    json_filename = "Sample_Title.json"
    all_data_articles_path.join(json_filename).write("")

    # Test file_finder function
    result = file_finder("file_123.json") # file_finder -> title_json: IndexError: index 0 is out of bounds for axis 0 with size 0
    expected_result = json_filename
    assert result == expected_result

def test_text_extraction(sample_article_dict):
    # Test TextExtraction class
    text_extraction = TextExtraction(sample_article_dict)
    
    # Test generate_context method
    context = text_extraction.generate_context(1)
    expected_context = "[CITATION-1]   \n   Footnote 1: Sample footnote content."
    assert context == expected_context
 
def test_text_extraction_footnote_mask(sample_article_dict):
    # Test TextExtraction class with footnote_mask set to False
    text_extraction = TextExtraction(sample_article_dict, footnote_mask=False)
    
    # Test generate_context method without footnote masking
    context = text_extraction.generate_context(1)
    expected_context = "[CITATION-1]   \n   Footnote 1: Sample footnote content."
    assert context == expected_context

def test_text_extraction_footnote_text(sample_article_dict):
    # Test TextExtraction class with footnote_text set to False
    text_extraction = TextExtraction(sample_article_dict, footnote_text=False)
    
    # Test generate_context method without footnote text
    context = text_extraction.generate_context(1)
    expected_context = "[CITATION-1]"
    assert context == expected_context

def test_text_extraction_previous_tokens(sample_article_dict):
    # Test TextExtraction class with previous_context_tokens set
    text_extraction = TextExtraction(sample_article_dict, previous_context_tokens=5)
    
    # Test generate_context method with previous_context_tokens
    context = text_extraction.generate_context(1)
    expected_context = "Sample article text [CITATION-1]   \n   Footnote 1: Sample footnote content."
    assert context == expected_context