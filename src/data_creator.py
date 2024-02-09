import json
from pathlib import Path
import pandas as pd
from .text_extraction import TextExtraction
from .utils import file_finder


def footnote_to_int(footnote_number) -> int:
    if not isinstance(footnote_number, int):
        try:
            footnote_number = int(footnote_number)
        except ValueError as exc:
            raise ValueError("Could not convert the value to an integer.") from exc
        except TypeError as exc:
            raise TypeError("The value is not a string or an integer.") from exc
    return footnote_number


def create_data(
    previous_context_tokens: int, following_context_tokens: int
) -> dict[pd.DataFrame]:
    """
    This function does the following things:
    1. It iterates over all annotated articel files:
       For each annotated article we create a pandas df, loop over all rows in its corresponding df object
    2. We extract generate the citation context and extract the corresponding footnote
    3. Context and footnotes are added to the df object

    Dependencies:
    - file_finder
    - TextExtraction
    """

    path_annotations = Path("./data/annotated")
    path_articles = Path("./all_data_articles")
    df_dict = {}
    for filepath in path_annotations.iterdir():
        df_name = filepath.name
        df = pd.read_excel(filepath)
        df_dict[df_name] = df
        title_json = file_finder(filepath.name)
        article_path = path_articles / title_json
        with open(article_path, "r", encoding="utf-8") as file:
            article_dict = json.load(file)
        contexts = []
        footnotes = []
        for i in range(len(df)):
            footnote_number = df["Footnote"].iloc[i]
            footnote_number = footnote_to_int(footnote_number)
            context = TextExtraction(
                article_dict,
                previous_context_tokens=previous_context_tokens,
                following_context_tokens=following_context_tokens,
                previous_context_sentences=None,
                following_context_sentences=None,
                previous_whole_paragraph=False,
                following_whole_paragraph=False,
                till_previous_citation=None,
                till_following_citation=None,
                footnote_text=False,
                footnote_mask=True,
            ).generate_context(footnote_number)
            contexts.append(context)
            footnote = article_dict["footnotes"][str(footnote_number)]
            footnotes.append(footnote)
        df["context"] = contexts
        df["footnote_text"] = footnotes
    return df_dict
