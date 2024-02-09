# testing ground

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.final_scraper import *


import time
import random
import json
import pytest
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_driver():
    # Mocking the WebDriver instance
    driver_mock = MagicMock()
    return driver_mock

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

def mock_random_sleep(*args, **kwargs):
    return 1.1

def test_scrape_footnotes(mock_driver):
    # Mocking the find_elements method to return citation elements       
    mock_citations = [MagicMock(text='1'), MagicMock(text='2')]
    mock_driver.find_elements.return_value = mock_citations

    # Mocking find_element to return references
    mock_references = [
        MagicMock(),  # First call in try block
        MagicMock(text='Reference 1'),  # Second call in try block       
        MagicMock(),  # First call in except block
        MagicMock(text='Reference 2'),  # Second call in except block    
    ]
    # Repeat the last element for any additional calls
    mock_driver.find_element.side_effect = mock_references + [mock_references[-1]] * 1

    # Debugging mock behavior
    print("Mocked find_elements result:", mock_driver.find_elements.return_value)
    print("Mocked find_element side effect:", mock_driver.find_element.side_effect)

    with patch('src.final_scraper.time.sleep', side_effect=mock_random_sleep):
        footnotes = scrape_footnotes(mock_driver)

    # Add assertions to verify the behavior of scrape_footnotes
    assert footnotes == {'1': 'Reference 1', '2': 'Reference 2'}