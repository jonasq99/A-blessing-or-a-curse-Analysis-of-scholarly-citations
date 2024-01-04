#import os
#import json
import pandas as pd
#from openai import OpenAI
#from text_extraction import file_finder, TextExtraction
#from dotenv import load_dotenv
#from pathlib import Path
from utils import get_completion, get_completion_from_messages, create_data


data = create_data()
data_keys = list(data.keys())
print(data_keys)

def get_sentiment(name, title, context, footnote):
    system_message = """
    You are an expert in analyzing citations from historical papers. It is your job 
    to determine if a citation is opinionated or neutral.
    """
    prompt = f"""
    You will receive the name of the author of the cited sourced, its title, the context 
    of the citation and its corresponding footnote. 
    The data will be submitted in the following format:
    #######################Begin format instructions####################################
    name: name of authors
    title: title of cited source
    context: The context of the citation
    footnote: The corresponding footnote text of the citation
    #######################End format instructions#################################
    In the context citations are annotated like this: "[Citation footnotenumber]".
    A citation counts as opinionated when the author makes a judgemental statement 
    about the cited source. This means the author rates its quality or 
    A citation is counted as neutral if it recites content from the source, or the author
    does not make an explicit statement about the quality of the work.
    Look closely at the text in the footnote! It can be the case that the hint if 
    a citation is neutral or opinionated might be located in the footnote text.
    Footnotes often contain multiple citations of different authors. 
    Look at the names of authors from the name field and relate them to the footnote. 
    Only rate the citation that is related to the name of authors in the name field. 
    Return 0 if a citation is neutral, return 1 when a citation is opinionated.
    Only return integerts 0 and 1, nothing else.
    #################Begin data##################################
    Data:
    name: {name}
    title: {title}
    context: {context}
    footnote: {footnote}
    ###############End data#########################################
    Your answer: Enter integer here
    """
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    prediction = get_completion_from_messages(messages)
    return prediction

df = data[data_keys[0]][0:10]

predictions = []
for i in range(len(df)):
    name = df["Authors"].iloc[i]
    title = df["Title"].iloc[i]
    context = df["context"].iloc[i]
    footnote = df["footnote_text"].iloc[i]

    pred = get_sentiment(name, title, context, footnote)
    print(pred)
    predictions.append(pred)
print(predictions)