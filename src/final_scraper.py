import json
import random
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def random_sleep():
    return random.uniform(1, 1.2)


def scrape_footnotes(driver):
    footnotes = {}
    citations = driver.find_elements(By.CSS_SELECTOR, "a.link-ref")
    for c in citations:
        label = c.text
        if label.isdigit():
            rev_id = "fn" + label.zfill(4)
            try:
                # Find the element with reveal-id="fn0001"
                element = driver.find_element(By.CSS_SELECTOR, f"[reveal-id={rev_id}]")

                if element:
                    # Scroll to the element using JavaScript
                    driver.execute_script(
                        "arguments[0].scrollIntoView({ behavior: 'smooth' });", element
                    )
            except Exception as e:
                print(f"Error: {e}")

            c.click()
            time.sleep(random_sleep())

            # Wait for the citation to be visible
            wait = WebDriverWait(driver, 10)
            wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, "p.footnote-compatibility")
                )
            )

            ref = driver.find_element(By.CSS_SELECTOR, "div#revealContent").text.split(
                "\n"
            )[-1]
            footnotes[label] = ref

            # Use JavaScript to click the "Close" button forcefully
            close_button = driver.find_element(By.CSS_SELECTOR, "a.close-reveal-modal")
            driver.execute_script("arguments[0].click();", close_button)

            time.sleep(random_sleep())
    return footnotes


def scrape_article(soup, url):
    paragraphs = soup.find_all("p", class_="chapter-para")

    # Initialize a variable to store the extracted text and citations
    result = ""

    # Iterate through paragraphs
    for paragraph in paragraphs:
        # Extract the text within the paragraph
        text = ""
        citations = []
        for element in paragraph.contents:
            if element.name == "a":
                # Extract the citation number from the sup tag
                try:
                    citation_number = int(element.find("sup").get_text())
                    # Add the citation to the list
                    citations.append(citation_number)

                except:
                    try:
                        citation_number = int(element.get_text())
                        citations.append(citation_number)

                    except:
                        citation_number = ""
                        print(element, url)

                if citation_number != "":
                    text += f"[CITATION-{citation_number}] "

            elif element and hasattr(element, "strip"):
                # Add the text content (if not None and has a strip method)
                try:
                    text += element.strip()
                except:
                    pass

        result += text

    return result, citations


def scrape_all(url):
    firefox_options = Options()
    firefox_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/100.0"
    )

    # Create a Firefox WebDriver instance
    driver = webdriver.Firefox(options=firefox_options)

    driver.get(url)

    time.sleep(random_sleep())
    accept_button = driver.find_element(By.XPATH, "//button[@id='accept-button']")
    accept_button.click()

    publication_data = {}

    # Get the page source after it has loaded
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    # Extract the title
    title_element = soup.find("h1", class_="wi-article-title")
    title = title_element.get_text(strip=True)
    publication_data["title"] = title

    # Extract the author
    author_element = soup.find("button", class_="linked-name")
    author = author_element.get_text(strip=True)
    publication_data["author"] = author

    # Extract the publication date
    date_element = soup.find("div", class_="citation-date")
    date = date_element.get_text(strip=True)
    publication_data["date"] = date

    article, footnotes_numbers = scrape_article(soup, url)
    publication_data["article"] = article

    footnotes = scrape_footnotes(driver)
    publication_data["footnotes"] = footnotes

    driver.close()

    return publication_data


def save_data(data):
    path = "all_data_articles/" + data["title"].replace(" ", "_") + ".json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved {path}")
