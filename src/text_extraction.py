import pandas as pd
import numpy as np 
import re
import os
import json
import nltk

def file_finder(file_str: str) -> str:
    """
    This function takes a file name and returns the path to the file in the all_data_articles.
    """
    title_doi = "../data/titles_doi.csv"
    folder_path = "../all_data_articles"
    
    #extract the doi from the file name
    doi = file_str.split("_")[-1].split(".")[0]

    # find the row in the csv file where the doi column ends with the doi
    df = pd.read_csv(title_doi)
    doi_row = df[df["DOI"].str.endswith(doi)]

    # extract the title from the row
    title_json = doi_row["Title"].values[0].replace(" ", "_") + ".json"

    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename.startswith(title_json[:int(len(title_json)/3)]):
            return filename
        
class TextExtraction:

    def __init__(self, article_dict: dict, previous_context_tokens: int = None, following_context_tokens: int = None, 
                 previous_context_sentences: int = None, following_context_sentences: int = None, previous_whole_paragraph: bool = None,
                 following_whole_paragraph: bool = None, till_previous_citation: int = None, till_following_citation: int = None,
                 footnote_text: bool = True, footnote_mask: bool = True):
        
        self.previous_context_tokens = previous_context_tokens
        self.following_context_tokens = following_context_tokens
        self.previous_context_sentences = previous_context_sentences
        self.following_context_sentences = following_context_sentences
        self.previous_whole_paragraph = previous_whole_paragraph
        self.following_whole_paragraph = following_whole_paragraph
        self.till_previous_citation = till_previous_citation
        self.till_following_citation = till_following_citation
        self.footnote_text = footnote_text
        self.footnote_mask = footnote_mask


        self.article_text = article_dict["article"]
        #keep in mind keys of this dict are strings of integers
        self.footnote_dict = article_dict["footnotes"]
    
    def generate_context(self, footnote_number: int):
        
        # Find the index of the footnote in the article text
        footnote_index = self.article_text.find(f"[CITATION-{footnote_number}]")

        if footnote_index == -1:
            raise ValueError(f"Footnote {footnote_number} not found in article text")
        
        # Get the content of the specified footnote
        footnote_content = self.footnote_dict[str(footnote_number)]

        # Extract the relevant context based on options
        start_index = max(0, self.find_previous_token_index(footnote_index))
        end_index = min(len(self.article_text), self.find_following_token_index(footnote_index + len(f"[CITATION-{footnote_number}]")))

        if self.previous_whole_paragraph:
            start_index = max(0, self.article_text.rfind('\n', 0, start_index) + 1)

        if self.following_whole_paragraph:
            end_index = self.article_text.find('\n', end_index)

        #TODO: maybe rework
        # Include n amount of previous sentences
        if self.previous_context_sentences:
            sentences = nltk.sent_tokenize(self.article_text[:start_index])
            start_index = max(0, start_index - sum(len(sentence) for sentence in sentences[-self.previous_context_sentences:]))
        
        #TODO: maybe rework
        # Include n amount of following sentences
        if self.following_context_sentences:
            sentences = nltk.sent_tokenize(self.article_text[end_index:])
            end_index = min(len(self.article_text), end_index + sum(len(sentence) for sentence in sentences[:self.following_context_sentences]))

        if self.till_previous_citation:
            if footnote_number - self.till_previous_citation < 0:
                start_index = 0
            else:    
                start_index = self.article_text.find(f"[CITATION-{footnote_number-self.till_previous_citation}]") + len(f"[CITATION-{footnote_number-self.till_previous_citation}]")

        if self.till_following_citation:
            end_index = self.article_text.find(f"[CITATION-{footnote_number+self.till_following_citation}]")

        context = self.article_text[start_index:end_index].strip()

        # Apply footnote mask if required
        if self.footnote_mask:
            context = self.replace_citations(context, footnote_number)


        # Add footnote text if required
        if self.footnote_text:
            context += '   \n   ' + f"Footnote {footnote_number}: {footnote_content}"

        return context
    
    @staticmethod
    def replace_citations(text: str, footnote_number: int) -> str:
        citation_pattern = r'\[CITATION-(\d+)\]'
        def replacer(match):
            if match.group(1) == str(footnote_number):
                return match.group(0)
            else:
                return "[MASK]" # TODO: clarify if we should use empty string instead
        replaced_text = re.sub(citation_pattern, replacer, text)
        return replaced_text


    def find_previous_token_index(self, index: int) -> int:
        if self.previous_context_tokens is None:
            return index
        
        count_tokens = 0
        while count_tokens < self.previous_context_tokens and index > 0:
            index -= 1
            if self.article_text[index].isspace():
                count_tokens += 1
        return index

    def find_following_token_index(self, index: int) -> int:
        if self.following_context_tokens is None:
            return index

        count_tokens = 0
        while count_tokens < self.following_context_tokens and index < len(self.article_text):
            index += 1
            if self.article_text[index-1].isspace():
                count_tokens += 1
                
        return index