import json
import math
import pandas as pd
from openai import OpenAI
from openai._types import NotGiven, NOT_GIVEN
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
    messages: str,
    model: str = "gpt-3.5-turbo-0125",  # "gpt-3.5-turbo-0125",  # "gpt-3.5-turbo",  # "gpt-4-0125-preview",
    temperature: int = 0,
    max_tokens=NOT_GIVEN,
    top_p=NOT_GIVEN,
    frequency_penalty=NOT_GIVEN,
    presence_penalty=NOT_GIVEN,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    return response.choices[0].message.content


def get_context(text: str) -> str:
    context = []
    tokens = text.split()
    for token in tokens:
        if "[MASK]" in token:
            context = []
        elif "CITATION" in token:
            context.append(token)
            break
        else:
            context.append(token)
    context = " ".join(context)
    return context


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
    - better, failed, argue, however, convincingy, nuanced, vague, fail, overlook, simplification, neglect

    A citation reproduces information if it does not make a statement about the quality of the cited work!

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


# def calculate_accuracy_per_label(predictions, labels, label_value):
# label_value_predictions = 0
# correct_predictions = 0
# for l, p in zip(labels, predictions):
# if l == label_value:
# label_value_predictions += 1
# if l == label_value and p == l:
# correct_predictions += 1
# return (
# 0
# if label_value_predictions == 0
# else correct_predictions / label_value_predictions
# )


def few_shot_cot(examples: str, citation: str, title: str, footnote: str) -> str:
    system_message = f"""
    You are receiving a citation that can be either neutral or opinionated.  
    A citation is opinionated if the author makes a statement about the quality of the cited work. 
    Otherwise it is neutral. You are going to receive a citation and a label. 
    A citation has a context, a title and a footnote. The citation in the input is marked with a 
    special token "[CITATION-footnotenumber]" at the end of the citation.

    The author makes a statement about the quality of the work if:
    - the author makes a judgemental statement about the quality of a  cited source. 
    - the author rates the quality of the work in a positive or negative mannser etc.
    Keywords of opinionated citations:
    - better, failed, argue, however, convincingy, nuanced, vague, fail, overlook, simplification, neglect

    A citation reproduces information and is therefor neutral if it does not make a statement about the quality of the cited work!


    Briefly explain why the citation you received is "neutral" or "opinionated" with a response length not exceeding 100 words. 
    Your answer should focus not on the content of the cited work but on the author's judgement of the cited work or the author of the cited work. 
    
    {examples}
    """

    prompt = f"""
    Citation: {citation}
    Title: {title}
    Footnote: {footnote}
    Label:
    Reasoning:
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    prediction = get_completion_from_messages(messages)
    return prediction


def get_fewshot_cot_examples(df: pd.DataFrame = None) -> str:

    if df is None:
        module_dir = (
            Path(__file__).resolve().parent
        )  # Get the directory of the current module
        csv_path = module_dir / "../data/few_shot_examples/fewshot_cot.csv"
        df = pd.read_csv(csv_path)

    few_shot_examples = ""
    for i in range(len(df)):
        try:
            math.isnan(df["input"][i])
        except:
            few_shot_examples += (
                df["input"][i].replace("\n\n", "\n")
                + "\nReaoning:\n"
                + df["output"][i].rstrip("\n")
                + "\n\n"
            )
    return few_shot_examples


def get_label(model_output: str) -> int:
    system_message = """
    You are going to receive a text and you have to determine if the model output is a neutral or opinionated citation.
    If the text specifies that the citation is neutral, return 0 and nothing else.
    If the text specifies that the citation is opinionated, return 1 and nothing else.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": model_output},
    ]
    label = get_completion_from_messages(messages)
    try:
        label = int(label)
        return label
    except ValueError:
        return label


def filter_label(dataframes_dict: dict[pd.DataFrame], label: int) -> pd.DataFrame:
    """
    Input:
        dictionary of pandas dataframes
        a label integer to filter the dataframes
    Output:
        a single pandas dataframe with all the rows with the specified label
    """
    filtered_dataframes = []

    for key, df in dataframes_dict.items():
        if "Label" in df.columns:
            filtered_df = df[df["Label"] == label]

            filtered_dataframes.append(filtered_df)

    result_df = pd.concat(filtered_dataframes, ignore_index=True)

    return result_df


def sample_data(dataframe: pd.DataFrame, num_rand_samples: int = 100) -> pd.DataFrame:
    """
    Samples a given dataframe and returns a new dataframe with the specified number of random samples.
    """
    sampled_df = dataframe.sample(n=num_rand_samples, random_state=42)
    return sampled_df


def calculate_accuracy_per_label(predictions, labels, label_value: int) -> float:
    """
    input:
        n-dim array of model predictions
        n-dim array of labels
        label_value: int specifying the label to calculate the accuracy for
    ourtput:
        float: accuracy for the specified label
    """
    # Create a boolean array indicating whether the label matches the specified value
    label_matches = [l == label_value for l in labels]

    # Extract predictions for instances where the label matches the specified value
    matched_predictions = [p for i, p in enumerate(predictions) if label_matches[i]]

    return (
        sum(matched_predictions) / 100
        if label_value == 1
        else (len(matched_predictions) - sum(matched_predictions)) / 100
    )
