#revise test_scrape_footnotes (runtime error), test_scrape_all (typeError)
# see comment on test_scrape_article

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


""" @pytest.fixture(autouse=True)
def mock_driver(mocker):
    # Mocking the webdriver.Firefox class
    #mock_driver = mocker.Mock(webdriver.Firefox)
    #return mock_driver
    return mocker.Mock(webdriver.Firefox) """

@pytest.fixture
def mock_driver():
    # Mocking the WebDriver instance
    driver_mock = MagicMock()
    return driver_mock

""" @pytest.fixture(autouse=True)
def mock_soup(mocker):
    # Mocking the BeautifulSoup class
    #mock_soup = mocker.Mock(BeautifulSoup)
    #return mock_soup
    return mocker.Mock(BeautifulSoup) """

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

def test_random_sleep():
    # Test random_sleep function
    sleep_time = random_sleep()
    assert 1 <= sleep_time <= 1.2

# Mocking the random_sleep function to return a constant value for testing
def mock_random_sleep(*args, **kwargs):
    return 1.1

# 2 verschiedene test functions, die beide runtime error anzeigen. Alle tipps von chatgpt führen letzlich zum runtime error
""" def test_scrape_footnotes(mock_driver, mocker):
    # Mocking driver.find_elements
    #mocker.patch.object(mock_driver, 'find_elements', return_value=[mocker.Mock(text='1'), mocker.Mock(text='2')])
    mocker.patch.object(mock_driver, 'find_elements', return_value=[MagicMock(text='1'), MagicMock(text='2')])
    
    #oder alternativ die function drunter für find element, ändert aber trotzdem nichts am runtime error
    # Mocking driver.find_element """
""" mocker.patch.object(mock_driver, 'find_element', side_effect=[
        mocker.Mock(),  # First call in try block
        mocker.Mock(text='Reference 1'),  # Second call in try block
        mocker.Mock(),  # First call in except block
        mocker.Mock(text='Reference 2'),  # Second call in except block
    ]) """

""" # Define a generator function to yield mock objects
    def find_element_mock(*args, **kwargs):
        # Define the mock objects to yield
        mock_objects = [
            mocker.Mock(text='Reference 1'),  # First call in try block
            mocker.Mock(text='Reference 2'),  # Second call in except block
            mocker.Mock(),  # Third call in try block
            mocker.Mock(),  # Fourth call in except block
            # Add more mock objects as needed for subsequent calls
        ]
        # Yield mock objects one by one
        for obj in mock_objects:
            yield obj

    # Mocking driver.find_element with side_effect
    mocker.patch.object(mock_driver, 'find_element', side_effect=find_element_mock())


    # Mocking time.sleep
    mocker.patch('time.sleep')

    # Mocking WebDriverWait
    mocker.patch('selenium.webdriver.support.ui.WebDriverWait.until')

    footnotes = scrape_footnotes(mock_driver)
    assert footnotes == {'1': 'Reference 1', '2': 'Reference 2'} """

""" def test_scrape_footnotes(mock_driver):
    # Mocking the find_elements method to return citation elements
    mock_citations = [MagicMock(text='1'), MagicMock(text='2')]
    mock_driver.find_elements.return_value = mock_citations

    # Mocking find_element to return references
    mock_driver.find_element.side_effect = [
        MagicMock(),  # First call in try block
        MagicMock(text='Reference 1'),  # Second call in try block
        MagicMock(),  # First call in except block
        MagicMock(text='Reference 2'),  # Second call in except block
    ]

    with patch('src.final_scraper.time.sleep', side_effect=mock_random_sleep):
        footnotes = scrape_footnotes(mock_driver)  
    assert footnotes == {'1': 'Reference 1', '2': 'Reference 2'} """

def test_scrape_article(mock_soup):
    # Call the function with the mock BeautifulSoup object
    result, citations = scrape_article(mock_soup, "example_url")
    
    # Define the expected result and citations
    expected_result = "[CITATION-1] This is some text with a citation.[CITATION-2] Another paragraph with a different citation."
    expected_citations = [2] # besser Liste aller citations und nicht nur letzten Loop durchlauf: dann [1, 2]
    #citations wird in jedem loop neu deklariert, wäre besser es vorher direkt auch bei result zu deklarieren

    # Compare the actual result and citations with the expected ones
    assert result == expected_result
    assert citations == expected_citations 

# 2 Funktionen für test scrape all
def test_scrape_all(mock_driver, mock_soup, mocker):
    # Mocking webdriver.Firefox
    mocker.patch('selenium.webdriver.Firefox', return_value=mock_driver)

    # Mocking driver.get
    mocker.patch.object(mock_driver, 'get')

    # Mocking time.sleep
    mocker.patch('time.sleep')

    # Mocking driver.find_element
    mocker.patch.object(mock_driver, 'find_element')

    #mock_page_source = "<html><h1>Title</h1><button>Accept</button><p>Content</p></html>"
    # Mocking BeautifulSoup
    #mocker.patch('bs4.BeautifulSoup', return_value=mock_soup)
    
    # Create a MagicMock object to mimic BeautifulSoup
    """ mock_bs = mocker.MagicMock(name='BeautifulSoup')

    # Configure mock_bs to behave like a BeautifulSoup instance
    mock_bs.return_value = mock_bs
    mock_bs.find_element.side_effect = lambda *args, **kwargs: mock_bs
    mock_bs.get_text.side_effect = lambda *args, **kwargs: 'Content'
    mocker.patch('bs4.BeautifulSoup', return_value=mock_bs) """

    # Patch bs4.BeautifulSoup with the MagicMock object
    mocker.patch('bs4.BeautifulSoup', return_value=mock_soup)

    # Mocking scrape_article and scrape_footnotes
    mocker.patch('src.final_scraper.scrape_article', return_value=("Article content", [1, 2]))
    mocker.patch('src.final_scraper.scrape_footnotes', return_value={'1': 'Reference 1', '2': 'Reference 2'})

    # Mocking save_data
    mocker.patch('src.final_scraper.save_data')

    url = "example_url"
    text = mock_soup.prettify()
    result = scrape_all(text)

    assert result == {
        'title': mocker.Mock(get_text=mocker.Mock(strip=mocker.Mock(return_value='Article Title'))),
        'author': mocker.Mock(get_text=mocker.Mock(strip=mocker.Mock(return_value='Author Name'))),
        'date': mocker.Mock(get_text=mocker.Mock(strip=mocker.Mock(return_value='Publication Date'))),
        'article': 'Article content',
        'footnotes': {'1': 'Reference 1', '2': 'Reference 2'}
    }

""" def test_scrape_all(mock_driver, mock_soup):
    mock_driver.page_source = '<html><h1 class="wi-article-title">Title</h1><button class="linked-name">Author</button><div class="citation-date">Date</div></html>'
    #mock_driver.find_element.return_value.text = "Accept"
    #mock_driver.get.return_value = None
    accept_button_mock = MagicMock()
    mock_driver.find_element.return_value = accept_button_mock

    expected_data = {
        'title': 'Title',
        'author': 'Author',
        'date': 'Date',
        'article': "[CITATION-1] This is some text with a citation.[CITATION-2] Another paragraph with a different citation.",
        'footnotes': {'1': 'Reference 1', '2': 'Reference 2'}
    }

    with patch('src.final_scraper.time.sleep', side_effect=mock_random_sleep):
        with patch('src.final_scraper.scrape_footnotes', return_value={'1': 'Reference 1', '2': 'Reference 2'}):
            data = scrape_all("http://example.com")
    
    assert data == expected_data
    mock_driver.find_element.assert_called_once_with(By.XPATH, "//button[@id='accept-button']")
    accept_button_mock.click.assert_called_once() """

def test_save_data(mocker):
    # Mocking json.dump
    mock_open = mocker.mock_open()
    mocker.patch('builtins.open', mock_open)
    mocker.patch('json.dump')

    data = {
        'title': 'Article_Title',
        'author': 'Author_Name',
        'date': 'Publication_Date',
        'article': 'Article content',
        'footnotes': {'1': 'Reference 1', '2': 'Reference 2'}
    }

    save_data(data)

    # Assuming your print statement is just for verification purposes
    expected_path = 'all_data_articles/Article_Title.json'
    assert mocker.call(expected_path, 'w') in mock_open.mock_calls