import json
import os
import sys
import warnings

import pandas as pd

from .fewshot_cot_classification import get_predictions
from .information_extraction import (
    information_extraction,
    extract_citations,
    tagger_information_extraction,
)
from .text_extraction import TextExtraction

warnings.filterwarnings("ignore")

if len(sys.argv) < 3:
    print(
        """
        Missing arguments:

        Usage: python main.py information_extraction_method filename
        
        information extraction method: anystyle, regex, tagger
        filename: the name of the file in the folder all_data_articles
    """
    )
    sys.exit(1)

information_extraction_method = sys.argv[1]
filename = sys.argv[2]

folder_path = "./all_data_articles"

"""
anystyle -> information_extraction
regex -> extract_citations
tagger -> tagger_information_extraction
"""

print("Information extraction method: ", information_extraction_method)
print("Filename: ", filename)
print("=====================================")
print("Extracting information...")

extraction = None

if information_extraction_method == "anystyle":
    extraction = information_extraction(filename)
elif information_extraction_method == "regex":
    extraction = extract_citations(filename)
elif information_extraction_method == "tagger":
    from flair.nn import Classifier

    tagger = Classifier.load("ner-ontonotes-large")
    extraction = tagger_information_extraction(filename, tagger)
else:
    raise ValueError("Invalid information_extraction_method")

extraction = sorted(list(extraction), key=lambda x: x[0])
print("Finished extracting information.")
print("=====================================")

with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
    article_dict = json.load(file)

previous_context_tokens = 500
following_context_tokens = 0

footnotes = article_dict["footnotes"]

text_extractor = TextExtraction(
    article_dict,
    previous_context_tokens=previous_context_tokens,
    following_context_tokens=following_context_tokens,
    footnote_text=False,
    footnote_mask=True,
)

previous_footnote = -1
previous_context = None

footnote_numbers = []
authors = []
titles = []
contexts = []
footnote_texts = []

for footnote_number, author, title in extraction:
    if int(footnote_number) != previous_footnote:
        context = text_extractor.generate_context(int(footnote_number))
    else:
        context = previous_context

    footnote_numbers.append(int(footnote_number))
    authors.append(author)
    titles.append(title)
    contexts.append(context)
    footnote_texts.append(footnotes[str(footnote_number)])

    previous_footnote = int(footnote_number)
    previous_context = context


few_shot_df = pd.DataFrame(columns=["Title", "context", "footnote_text"])

few_shot_df["Title"] = titles
few_shot_df["context"] = contexts
few_shot_df["footnote_text"] = footnote_texts

print("Starting predictions")

predictions = get_predictions(few_shot_df)
print("Finished predictions")
print("=====================================")

result_df = pd.DataFrame(columns=["footnote", "author", "title", "label"])

result_df["footnote"] = footnote_numbers
result_df["author"] = authors
result_df["title"] = titles
result_df["label"] = predictions

result_folder = os.path.join(os.path.join("./results", information_extraction_method))
if not os.path.exists(result_folder):
    os.makedirs(result_folder, exist_ok=True)

result_df.to_csv(
    os.path.join(
        "./results", information_extraction_method, filename.replace(".json", ".csv")
    ),
    index=False,
)
print(f"Stored result file in {result_folder}")
