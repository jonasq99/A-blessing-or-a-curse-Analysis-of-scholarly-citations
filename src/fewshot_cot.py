import json
import pandas as pd
from data_creator import create_data
from utils import (
    get_context,
    few_shot_cot,
    get_fewshot_cot_examples,
    get_parseable_json_from_string,
    filter_label,
    sample_data,
    calculate_accuracy_per_label,
)
from dotenv import load_dotenv
import logging
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def get_precictions(df, failed_prection_counter: int = 3):
    predictions = []
    for i in range(len(df)):
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

        pred = get_parseable_json_from_string(pred)

        if pred is False:
            for i in range(failed_prection_counter):
                pred = few_shot_cot(
                    examples=get_fewshot_cot_examples(),
                    citation=context,
                    title=title,
                    footnote=footnote,
                )
                pred = get_parseable_json_from_string(pred)
                if pred:
                    break

        predictions.append(pred)
        # pred = get_label(pred)
        print(pred)
        predictions.append(pred)
        # predictions = [int(i) for i in predictions]
    return predictions


df_dict = create_data(500, 0)

opinionated_data = filter_label(df_dict, 1)
neutral_data = sample_data(filter_label(df_dict, 0))

df = pd.concat([opinionated_data, neutral_data], ignore_index=True)

y_pred = get_precictions(df)
