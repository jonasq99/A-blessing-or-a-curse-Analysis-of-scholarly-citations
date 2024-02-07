import json
import pandas as pd
from data_creator import create_data
from utils import (
    get_context,
    few_shot_cot,
    get_fewshot_cot_examples,
    get_label,
    filter_label,
    sample_data,
    calculate_accuracy_per_label,
)
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm


def get_precictions(df):
    predictions = []
    for i in tqdm(range(len(df))):
        # name = df["Authors"].iloc[i]
        title = df["Title"].iloc[i]
        context = get_context(df["context"].iloc[i])
        footnote = df["footnote_text"].iloc[i]
        # pred = context_sentiment(context)
        pred = few_shot_cot(
            examples=get_fewshot_cot_examples(),
            citation=context,
            title=title,
            footnote=footnote,
        )

        # while pred != "0" and pred != "1":
        # print("Retrying prediction...")
        # pred = few_shot_cot(examples = get_fewshot_cot_examples(), citation=context, title=title, footnote=footnote)
        pred = get_label(pred)
        predictions.append(pred)
        # predictions = [int(i) for i in predictions]
    return predictions


df_dict = create_data(500, 0)

opinionated_data = filter_label(df_dict, 1)
neutral_data = sample_data(filter_label(df_dict, 0))

df = pd.concat([opinionated_data, neutral_data], ignore_index=True)

y_pred = get_precictions(df)
y_true = df["Label"].tolist()

f1 = f1_score(y_pred, y_true)
accuracy_label_0 = calculate_accuracy_per_label(y_pred, y_true, label_value=0)
accuracy_label_1 = calculate_accuracy_per_label(y_pred, y_true, label_value=1)

print("F1 score:", f1)
print("Accuracy for label 0:", accuracy_label_0)
print("Accuracy for label 1:", accuracy_label_1)
