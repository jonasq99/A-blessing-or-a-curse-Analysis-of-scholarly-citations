import pandas as pd
import json
import os
import subprocess
import bibtexparser
from tqdm import tqdm
import re
from flair.data import Sentence
from fuzzywuzzy import fuzz

SPLITT_PATTERN = re.compile("; see |; | . See also | .See also |. See |, see")


def load_annotations(file_str: str) -> pd.DataFrame:
    """
    This function takes a file name and returns the annotations from the file.
    And also replaces missing values with None.
    """
    folder_path = "./data/annotated"

    file_path = os.path.join(folder_path, file_str)
    df = pd.read_excel(file_path)

    # replace missing values with None
    df = df.where(pd.notnull(df), None)

    # replace values marked with nan with None
    df = df.replace("nan", None)

    return df


def format_author_name(name):
    if name is None:
        return None
    if " and " in name:
        # Handle multiple authors
        authors = name.split(" and ")
        formatted_authors = [format_author_name(author) for author in authors]
        return " and ".join(formatted_authors)
    else:
        parts = name.split()
        # Handle case where there is a middle initial
        if len(parts) == 3:
            return f"{parts[1]}, {parts[0]} {parts[2]}"
        # Handle case where there is no middle initial
        elif len(parts) == 2:
            return f"{parts[1]}, {parts[0]}"
        else:
            return name


def df_to_triplets(df: pd.DataFrame, format_author=True) -> set:
    """
    This function takes a dataframe and returns a set of triplets.
    """
    triplets = set()
    for i in range(len(df)):
        if format_author:
            triplet = (
                df.iloc[i]["Footnote"],
                format_author_name(df.iloc[i]["Authors"]),
                df.iloc[i]["Title"],
            )
        else:
            triplet = (
                df.iloc[i]["Footnote"],
                df.iloc[i]["Authors"],
                df.iloc[i]["Title"],
            )
        triplets.add(triplet)
    return triplets


def dict_to_triplets(extraction: dict) -> set:
    """
    Converts a dictionary of footnotes to a set of triplets
    """
    triplets = set()

    for number, references in extraction.items():
        for reference in references:
            author = reference[0]
            title = reference[1]

            if author == "":
                author = None

            if title == "":
                title = None

            triplets.add((int(number), author, title))

    return triplets


def information_extraction(
    file_path: str, path="./all_data_articles", path_to_anystyle="."
) -> set:
    """
    This function takes a file path and returns a set of triplets.
    """
    file_path = os.path.join(path, file_path)
    article = json.load(open(file_path, "r"))
    extraction = {}

    prev_footnote = None

    for number, footnote in tqdm(article["footnotes"].items()):

        # If the footnote is ibid, use the previous footnote
        if footnote.startswith("Ibid"):
            # do not fully replace ibid with previous footnote but rather prepend it since there might be other references after ibid
            footnote = prev_footnote + "; " + footnote.lstrip("Ibid. ")

        # Store the footnote for the next iteration
        prev_footnote = footnote

        references = SPLITT_PATTERN.split(footnote)

        author_title_list = []

        for reference in references:

            command = [
                "ruby",
                os.path.join(path_to_anystyle, "anystyle.rb"),
                str(reference).strip(),
            ]
            bibtex = subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout
            parsed_bibtex = bibtexparser.loads(bibtex).entries

            if parsed_bibtex:
                parsed_bibtex = parsed_bibtex[0]
            else:
                # print(f"No valid BibTeX entry found in: {bibtex}, set to empty dict")
                parsed_bibtex = {}

            if "note" in parsed_bibtex:
                continue

            # Extract title and author
            title = parsed_bibtex.get("title", parsed_bibtex.get("booktitle", None))
            author = parsed_bibtex.get("author", parsed_bibtex.get("editor", None))

            if author is not None or title is not None:
                # Append author and title pair to the list
                author_title_list.append([author, title])

        # Store the list in the extraction dictionary with the footnote number as the key
        extraction[number] = author_title_list

    return dict_to_triplets(extraction)


def calculate_scores(triplets, extractions):
    TP = len(triplets & extractions)  # Intersection of triplets and extractions
    FP = len(extractions - triplets)  # Elements in extractions but not in triplets
    FN = len(triplets - extractions)  # Elements in triplets but not in extractions

    recall = TP / (TP + FN) if TP + FN != 0 else 0
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    f_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )

    return recall, precision, f_score


def calculate_similarity(str1, str2):
    # return SequenceMatcher(None, str1, str2).ratio()

    # https://pypi.org/project/fuzzywuzzy/
    return fuzz.token_set_ratio(str1, str2) / 100


def evaluate_extraction(set1, set2, threshold=0.95):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for triplet1 in set1:
        footnote1, author1, title1 = triplet1
        author1 = author1 if author1 is not None else ""
        title1 = title1 if title1 is not None else ""
        concat_str1 = str(author1) + " " + str(title1)
        found_match = False

        for triplet2 in set2:
            footnote2, author2, title2 = triplet2
            author2 = author2 if author2 is not None else ""
            title2 = title2 if title2 is not None else ""
            concat_str2 = str(author2) + " " + str(title2)

            # Check for footnote number and similarity
            if (
                footnote1 == footnote2
                and calculate_similarity(concat_str1, concat_str2) >= threshold
            ):
                found_match = True
                break

        if found_match:
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = len(set2) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f_score



def extract_citations(file_path: str, path="./all_data_articles") -> set:
    file_path = os.path.join(path, file_path)
    article = json.load(open(file_path, "r"))
    citations = set()
    prev_footnote = None

    for footnote_number, footnote_text in tqdm(article["footnotes"].items()):
        # If the footnote is ibid, use the previous footnote
        if footnote_text.startswith("Ibid"):
            footnote_text = prev_footnote

        prev_footnote = footnote_text

        # Split the footnote into individual citations
        individual_citations = SPLITT_PATTERN.split(footnote_text)

        for citation_text in individual_citations:
            # Regular expression to extract authors and titles
            # TODO: try a better pattern
            pattern = re.compile(r"^(.+?),\s+(.+?)[,|(]")

            match = pattern.match(citation_text)

            if match:
                author = match.group(1)
                title = match.group(2)
                citations.add((int(footnote_number), author, title))
    return citations


def tagger_information_extraction(
    file_path: str, tagger, path="./all_data_articles"
) -> set:
    file_path = os.path.join(path, file_path)
    article = json.load(open(file_path, "r"))

    citations = set()
    prev_footnote = None

    for footnote_number, footnote_text in tqdm(article["footnotes"].items()):
        # If the footnote is ibid, use the previous footnote
        if footnote_text.startswith("Ibid")
            footnote_text = prev_footnote

        prev_footnote = footnote_text

        # Split the footnote into individual citations
        individual_citations = SPLITT_PATTERN.split(footnote_text)

        for citation_text in individual_citations:

            author = None

            sentence = Sentence(citation_text)
            tagger.predict(sentence)
            for span in sentence.get_spans("ner"):
                if span.tag == "PERSON" or span.tag == "ORG":
                    if author is None:
                        author = span.text
                    else:
                        author += "and " + span.text
                if span.tag == "WORK_OF_ART":
                    citations.add((int(footnote_number), author, span.text))
                    author = None

    return citations
