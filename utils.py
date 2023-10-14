# import time
from typing import Dict, List, TypeVar
import requests
from bs4 import BeautifulSoup as bs

BS = TypeVar('BS', bound=bs)

def get_footnotes(soup: BS) -> Dict[str, str]:
    """
    Extracts footnotes from a given Beautiful Soup element of articles from 
    https://academic.oup.com/ehr/search-results?f_ContentSubTypeDisplayName=Research+Article&fl_SiteID=5158&access_all=true&page=.
    
    :param soup: A Beautiful Soup element containing the HTML content of an article link.
    :return: A dictionary where each key is a footnote label (e.g. "1", "2", "3", etc.) and each value is the corresponding footnote content as a string.
    :raises ValueError: If the length of the footnote labels and content do not match.
    """
    footnote_labels: List[BS] = soup.find_all('span', attrs={'class': 'label fn-label'})
    footnote_contents: List[BS] = soup.find_all('div', attrs={'class': 'footnote-content'})
    if len(footnote_labels) != len(footnote_contents):
        raise ValueError('footnote_labels and footnote_contents have different lengths')
    footnotes: Dict[str, str] = {label.get_text(strip=True): content.get_text(strip=True) for label, content in zip(footnote_labels, footnote_contents)}
    return footnotes

def test_run_footnotes():
    """
    This is just a function for testing the behavior of the get_footnotes function
    """
    URL = 'https://doi.org/10.1093/ehr/cead151'
    # using a fake user agent so the website doesn't identify us as a web crawler
    HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}
    response = requests.get(URL, headers = HEADERS)
    soup = bs(response.text, features = 'html.parser')
    footnotes_dict = get_footnotes(soup)
    print(footnotes_dict)

if __name__ == '__main__':
    test_run_footnotes()