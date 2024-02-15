import json
from datetime import datetime
import pandas as pd
from .data_creator import create_data
from .utils import (
    get_context,
    few_shot_cot,
    get_fewshot_cot_examples,
    filter_label,
    sample_data,
    calculate_accuracy_per_label,
    llm_label_parser,
)
import logging
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import trange

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def results_to_json(metrics: dict[float], description: str = None, path: str = None):
    if path is None:
        path = Path(
            f"./experiments/fewshot_cot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )

    if description is None:
        description = "Fewshot COT evaluation on 100 opinionated samples and 100 random neutral samples from the test set"

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {"description": description, "datetime": current_time, "metrics": metrics}

    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


def get_predictions(df: pd.DataFrame, failed_prection_counter: int = 3) -> list[int]:
    predictions = []
    for i in trange(len(df)):
        title = df["Title"].iloc[i]
        context = get_context(df["context"].iloc[i])
        footnote = df["footnote_text"].iloc[i]

        pred = few_shot_cot(
            examples=get_fewshot_cot_examples(),
            citation=context,
            title=title,
            footnote=footnote,
        )

        pred = llm_label_parser(pred)
        if pred is None:
            for i in range(failed_prection_counter):
                pred = few_shot_cot(
                    examples=get_fewshot_cot_examples(),
                    citation=context,
                    title=title,
                    footnote=footnote,
                )
                if pred:
                    break

        predictions.append(pred)
    return predictions


if __name__ == "__main__":

    df_dict = create_data(500, 0)

    opinionated_data = filter_label(df_dict, 1)
    neutral_data = sample_data(filter_label(df_dict, 0))

    df = pd.concat([opinionated_data, neutral_data], ignore_index=True)

    y_pred = get_predictions(df)
    df.to_csv(
        f"./experiments/fewshot_cot_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        index=False,
    )
    y_true = df["Label"].tolist()

    f1_opinionated = f1_score(y_pred, y_true, pos_label=1)
    f1_neural = f1_score(y_pred, y_true, pos_label=0)
    accuracy_label_0 = calculate_accuracy_per_label(y_pred, y_true, label_value=0)
    accuracy_label_1 = calculate_accuracy_per_label(y_pred, y_true, label_value=1)

    metrics = {
        "neutral": {
            "f1": f1_neural,
            "recall": accuracy_label_0[0],
            "precision": accuracy_label_0[1],
        },
        "opinionated": {
            "f1": f1_opinionated,
            "recall": accuracy_label_1[0],
            "precision": accuracy_label_1[1],
        },
    }

    results_to_json(metrics)
