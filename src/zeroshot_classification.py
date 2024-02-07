import json
from datetime import datetime
import pandas as pd
from data_creator import create_data
from utils import (
    get_context,
    zero_shot,
    filter_label,
    sample_data,
    calculate_accuracy_per_label,
)
import logging
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import trange


LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def results_to_json_zs(metrics: dict[float], description: str = None, path: str = None):
    if path is None:
        path = Path(
            f"./experiments/zeroshot_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"
        )

    if description is None:
        description = "Zeroshot evaluation on 100 opinionated samples and 100 random neutral samples from the test set"

    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    data = {"description": description, "datetime": current_time, "metrics": metrics}

    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


def get_precictions_zs(df: pd.DataFrame, failed_prection_counter: int = 3) -> list[int]:
    predictions = []
    for i in trange(len(df)):
        name = df["Authors"].iloc[i]
        title = df["Title"].iloc[i]
        context = get_context(df["context"].iloc[i])
        footnote = df["footnote_text"].iloc[i]

        pred = zero_shot(name, title, context, footnote)

        if pred != "0" and pred != "1":
            for i in range(failed_prection_counter):
                pred = zero_shot(name, title, context, footnote)
                if pred == "0" or pred == "1":
                    break
        if pred != "0" and pred != "1":
            pred = None

        predictions.append(pred)
        predictions = [int(i) for i in predictions]

    return predictions


if __name__ == "__main__":

    df_dict = create_data(500, 0)

    opinionated_data = filter_label(df_dict, 1)
    neutral_data = sample_data(filter_label(df_dict, 0))

    df = pd.concat([opinionated_data, neutral_data], ignore_index=True)

    y_pred = get_precictions_zs(df)
    y_true = df["Label"].tolist()

    f1 = f1_score(y_pred, y_true)
    accuracy_label_0 = calculate_accuracy_per_label(y_pred, y_true, label_value=0)
    accuracy_label_1 = calculate_accuracy_per_label(y_pred, y_true, label_value=1)

    metrics = {
        "f1": f1,
        "acc neutral": accuracy_label_0,
        "acc opinionated": accuracy_label_1,
    }

    results_to_json_zs(metrics)
