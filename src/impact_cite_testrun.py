import json
import os
from datetime import datetime
from sklearn.metrics import f1_score
import pandas as pd
import torch
from datasets import DatasetDict, Dataset, concatenate_datasets
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import logging
from pathlib import Path
from tqdm import tqdm
from .data_creator import create_data
from .utils import calculate_accuracy_per_label, filter_label, get_context, sample_data

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

model = AutoModelForSequenceClassification.from_pretrained(
    "Velkymoss/impact-cite_v0.11", num_labels=2, token=token
)
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
model.eval()


def tokenize_seqs(examples):
    return tokenizer(
        examples["citation"], padding=True, truncation=True, return_tensors="pt"
    )


def run(
    preceeding_context: int,
    suceeding_context: int,
    citation_only: bool = True,
    small_test_set: bool = True,
) -> dict:
    df_dict = create_data(preceeding_context, suceeding_context)

    if small_test_set:
        opinionated_data = filter_label(df_dict, 1)
        neutral_data = sample_data(filter_label(df_dict, 0))
        df_dict = {"opinionated": opinionated_data, "neutral": neutral_data}

    # concatenate context and footnote text, select relevant columns
    for d in df_dict:
        if citation_only:
            df_dict[d]["context"] = df_dict[d]["context"].apply(get_context)

        df_dict[d]["citation"] = (
            df_dict[d]["context"] + " [Footnote] " + df_dict[d]["footnote_text"]
        )
        df_dict[d] = df_dict[d].loc[:, ["Label", "citation"]]
        df_dict[d] = Dataset.from_pandas(df_dict[d])
    # convert to HuggingFace Dataset
    dataset = DatasetDict(df_dict)

    # tokenize data
    test_data = dataset.map(tokenize_seqs, batched=True)
    test_data = test_data.rename_column("Label", "labels")
    test_data = concatenate_datasets(test_data.values())

    labels = test_data["labels"]

    predictions = []
    for i in tqdm(range(len(test_data)), desc="Predicting"):
        with torch.no_grad():
            logits = model(torch.tensor([test_data["input_ids"][i]])).logits
        pred = torch.argmax(logits).item()
        predictions.append(pred)

    f1_opinionated = f1_score(predictions, labels, pos_label=1)
    f1_neural = f1_score(predictions, labels, pos_label=0)
    accuracy_label_0 = calculate_accuracy_per_label(predictions, labels, label_value=0)
    accuracy_label_1 = calculate_accuracy_per_label(predictions, labels, label_value=1)

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

    return metrics, predictions


def results_to_json(metrics: dict, description: str = None, path: str = None):
    if path is None:
        path = Path(
            f"./experiments/impact_cite_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )

    if description is None:
        description = "ImpactCite evaluation on test set"

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {"description": description, "datetime": current_time, "metrics": metrics}

    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    results, predictions = run(500, 0)
    df = pd.DataFrame(predictions)
    df.to_csv(
        f"./experiments/impact_cite_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        index=False,
    )
    results_to_json(
        results,
        "ImpactCite evaluation on 100 opinionated and 100 neutral samples from test set",
    )
