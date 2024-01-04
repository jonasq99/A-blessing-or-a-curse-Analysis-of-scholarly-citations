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
    In the context citations are annotated like this: "[Citation footnotenumber]".

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

for i in range(len(data_keys)):
    df = data[data_keys[i]][0:10]

    predictions = []
    for i in range(len(df)):
        name = df["Authors"].iloc[i]
        title = df["Title"].iloc[i]
        context = df["context"].iloc[i]
        footnote = df["footnote_text"].iloc[i]

        pred = get_sentiment(name, title, context, footnote)
        predictions.append(pred)
    print(predictions)
