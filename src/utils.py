import json
import pandas as pd
from openai import OpenAI
from text_extraction import file_finder, TextExtraction
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI()


def get_completion(prompt:str, model:str="gpt-3.5-turbo")->str:
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, #
    )
    return response.choices[0].message.content


def get_completion_from_messages(messages:str, model:str="gpt-3.5-turbo", temperature:int=0)->str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def create_data()->dict[pd.DataFrame]:
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
    path_annotations = Path("../data/annotated")
    path_articles = Path("../all_data_articles")

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
            footnote_number = int(df['Footnote'].iloc[i])
            if not isinstance(footnote_number, int):
                try:
                    footnote_number = int(footnote_number)
                except ValueError as exc:
                    raise ValueError("Could not convert the value to an integer.") from exc
            context = TextExtraction(article_dict, previous_context_tokens=45, following_context_tokens=45,
                        previous_context_sentences=None, following_context_sentences=None,
                        previous_whole_paragraph=False, following_whole_paragraph=False,
                        till_previous_citation=None, till_following_citation=None
                    , footnote_text=False, footnote_mask=True
                    ).generate_context(footnote_number)
            contexts.append(context)
            footnote = article_dict["footnotes"][str(footnote_number)]
            footnotes.append(footnote)
        df["context"] = contexts
        df["footnote_text"] = footnotes

    return df_dict

