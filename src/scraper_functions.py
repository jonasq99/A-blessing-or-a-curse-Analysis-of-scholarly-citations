import time
import json
import pickle
import requests
from bs4 import BeautifulSoup as bs


def get_footnotes(soup: bs) -> dict[str, str]:
    """
    Extracts footnotes from a given Beautiful Soup element of articles from 
    https://academic.oup.com/ehr/search-results?f_ContentSubTypeDisplayName=Research+Article&fl_SiteID=5158&access_all=true&page=.
    
    :param soup: A Beautiful Soup element containing the HTML content of an article link.
    :return: A dictionary where each key is a footnote label (e.g. "1", "2", "3", etc.) and each value is the corresponding footnote content as a string.
    :raises ValueError: If the length of the footnote labels and content do not match.
    """
    footnote_labels: list[bs] = soup.find_all('span', attrs={'class': 'label fn-label'})
    footnote_contents: list[bs] = soup.find_all('div', attrs={'class': 'footnote-content'})
    if len(footnote_labels) != len(footnote_contents):
        raise ValueError('footnote_labels and footnote_contents have different lengths')
    footnotes: dict[str, str] = {label.get_text(strip=True): content.get_text(strip=True) for label, content in
                                 zip(footnote_labels, footnote_contents)}
    return footnotes


def get_soup_elements(url: str, headers: dict[str, str] = None) -> bs:
    """
    Fetches a webpage and returns a BeautifulSoup object for its HTML content.

    :param url: The URL of the webpage to fetch.
    :param headers: Optional HTTP headers for the request (default: None).

    :return: A BeautifulSoup object representing the webpage's HTML content.

    :raises: Exception if an HTTP error occurs during the request.
    """
    if headers is None:
        # using a fake user agent so the website doesn't identify us as a web crawler
        headers: dict[str, str] = {
            'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}
    try:
        response: requests.models.Response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup: bs = bs(response.text, features='html.parser')
        return soup
    except requests.exceptions.HTTPError as error:
        raise Exception(f"An error occurred during the request: {error}")


def scraper(links: list[str], pause: int = 10, write_to_file: bool = False,
            filepath: str = "../data/article_soup_data.pkl") -> list[bs]:
    """
    Scrapes web pages, extracts HTML content, and optionally saves it to a file.

    :param links: List of URLs to scrape.
    :param pause: Time to wait between failed requests (default: 10 seconds).
    :param write_to_file: Flag to save data to a file (default: False).
    :param filepath: Filepath for saving data (default: "data/article_soup_data.pkl").

    :return: List of BeautifulSoup objects containing HTML content.
    """
    soup_data: list[bs] = []
    idx: int = 0
    while idx < len(links):
        try:
            soup: bs = get_soup_elements(links[idx])
            soup_data.append(soup)
            idx += 1
        except:
            time.sleep(pause)
            pause += 10
    if write_to_file is True:
        with open(filepath, "wb") as file:
            pickle.dump(soup_data, file)
    return soup_data


def load_soupdata(filepath: str = "../data/article_soup_data.pkl") -> list[bs]:
    """
    Load previously saved BeautifulSoup data from a file.

    :param filepath: Filepath for the saved data (default: "data/article_soup_data.pkl").
    
    :return: List of BeautifulSoup objects containing HTML content.
    """
    with open(filepath, "rb") as file:
        loaded_soup: list[bs] = pickle.load(file)
    return loaded_soup


def load_article_data(filepath: str = "../data/article_data.json") -> list[bs]:
    with open(filepath, 'r') as json_file:
        loaded_serialized_list = json.load(json_file)
        loaded_soup_list = [bs(item, 'html.parser') for item in loaded_serialized_list]
    return loaded_soup_list
