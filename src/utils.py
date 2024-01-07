import json
import pandas as pd
from openai import OpenAI
from text_extraction import file_finder, TextExtraction
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI()


def get_completion(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content


def get_completion_from_messages(
    messages: str, model: str = "gpt-3.5-turbo", temperature: int = 0
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


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
            footnote_number = int(df["Footnote"].iloc[i])
            if not isinstance(footnote_number, int):
                try:
                    footnote_number = int(footnote_number)
                except ValueError as exc:
                    raise ValueError(
                        "Could not convert the value to an integer."
                    ) from exc
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


def zero_shot(name: str, title: str, context: str, footnote: str) -> str:
    system_message = """
    You are an expert in analyzing citations from historical papers. It is your job 
    to determine if the author makes a statement about the quality of the work or just 
    uses it for the purpose of information reproduction.
    """
    prompt = f"""
    You will receive the name of the author of the cited source, its title, the context 
    of the citation and its corresponding footnote. 
    The data will be submitted in the following format:
    #######################Begin format instructions####################################
    name: name of authors
    title: title of cited source
    context: The context of the citation
    footnote: The corresponding footnote text of the citation
    #######################End format instructions#################################
    In the context citations are annotated like this: "[CITATION-footnotenumber]".

    The author makes a statement about the quality of the work if:
    - the author makes a judgemental statement about the quality of a  cited source. 
    - the author rates the quality of the work in a positive or negative mannser etc.
    Keywords of opinionated citations:
    - better, failed, argue, however, convincingy, nuanced, vague, fail, overlook, simplification
    Be very strict when labeling if a statement about the quality of a work is made. Only do so if the criteria match precisely.

    A citation reproduces information if it does not make an explicit statement about the quality of the cited work!

    Look closely at the text in the footnote! It can be the case that the hint if 
    a citation is neutral or opinionated might be located in the footnote text.
    Footnotes often contain multiple citations of different authors. 
    Look at the names of authors from the name field and relate them to the footnote. 
    Only rate the citation that is related to the name of authors in the name field. 

    Return 1 if the author makes a statement about the quality of the work else 0.
    Only return integers 0 and 1, nothing else.
    #################Begin data##################################
    Data:
    name: {name}
    title: {title}
    context: {context}
    footnote: {footnote}
    ###############End data#########################################
    Your answer: Enter integer here
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    prediction = get_completion_from_messages(messages)
    return prediction
