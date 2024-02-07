## all test pass

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.scraper_functions import *

import pytest
from bs4 import BeautifulSoup

@pytest.fixture
def sample_soup():
    html_content = """
    <html>
        <body>
            <span class="label fn-label">1</span>
            <div class="footnote-content">Footnote 1 content</div>
            <span class="label fn-label">2</span>
            <div class="footnote-content">Footnote 2 content</div>
        </body>
    </html>
    """
    return BeautifulSoup(html_content, 'html.parser')

def test_get_footnotes(sample_soup):
    footnotes = get_footnotes(sample_soup)
    expected_footnotes = {'1': 'Footnote 1 content', '2': 'Footnote 2 content'}
    assert footnotes == expected_footnotes

def test_get_soup_elements(requests_mock):
    url = 'https://example.com'
    content = '<html><body>Sample content</body></html>'
    requests_mock.get(url, text=content)

    soup = get_soup_elements(url)
    assert isinstance(soup, BeautifulSoup)
    assert soup.get_text() == 'Sample content'

def test_scraper(requests_mock, tmp_path):
    url1 = 'https://example.com/1'
    url2 = 'https://example.com/2'
    content = '<html><body>Sample content</body></html>'
    requests_mock.get(url1, text=content)
    requests_mock.get(url2, text=content)

    file_path = tmp_path / 'article_soup_data.pkl'
    links = [url1, url2]

    soup_data = scraper(links, pause=0, write_to_file=True, filepath=file_path)
    assert isinstance(soup_data, list)
    assert len(soup_data) == 2

    loaded_soup_data = load_soupdata(filepath=file_path)
    assert isinstance(loaded_soup_data, list)
    assert len(loaded_soup_data) == 2

def test_load_article_data(tmp_path):
    # Assuming you have a JSON file with serialized HTML content
    json_content = '["<html><body>Article 1 content</body></html>", "<html><body>Article 2 content</body></html>"]'
    json_file = tmp_path / 'article_data.json'
    json_file.write_text(json_content)

    loaded_soup_list = load_article_data(filepath=json_file)
    assert isinstance(loaded_soup_list, list)
    assert len(loaded_soup_list) == 2
    assert all(isinstance(soup, BeautifulSoup) for soup in loaded_soup_list)