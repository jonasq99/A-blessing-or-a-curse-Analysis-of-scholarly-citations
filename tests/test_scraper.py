from src.scraper import scrape_footnotes, scrape_article, save_data

import pytest
from bs4 import BeautifulSoup
from unittest.mock import MagicMock, patch

# TODO: Write unit tests for the scrape_footnotes, scrape_article, and save_data functions
# test for scrape_all can be done as well but might not be necessary

@pytest.fixture
def mock_soup():
    # Create a mock BeautifulSoup object
    html_content = """
    <html>
    <body>
        <p class="chapter-para">
            <a><sup>1</sup></a>
            This is some text with a citation.
        </p>
        <p class="chapter-para">
            <a><sup>2</sup></a>
            Another paragraph with a different citation.
        </p>
    </body>
    </html>
    """
    soup = BeautifulSoup(html_content, "html.parser")
    return soup

@pytest.mark.correct
def test_scrape_article(mock_soup):
    # Call the function with the mock BeautifulSoup object
    result, citations = scrape_article(mock_soup, "example_url")
    
    # Define the expected result and citations
    expected_result = "[CITATION-1] This is some text with a citation.[CITATION-2] Another paragraph with a different citation."
    expected_citations = [1, 2] # or [1], depends on how to put citation variable in 

    # Compare the actual result and citations with the expected ones
    assert result == expected_result
    assert citations == expected_citations 
