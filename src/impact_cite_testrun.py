import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from utils import calculate_accuracy_per_label, create_data

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


def run(preceeding_context: int, suceeding_context: int):
    df_dict = create_data(preceeding_context, suceeding_context)

    # concatenate context and footnote text, select relevant columns
    for d in df_dict:
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
    for i in range(len(test_data)):
        with torch.no_grad():
            logits = model(torch.tensor([test_data["input_ids"][i]])).logits
        pred = torch.argmax(logits).item()
        predictions.append(pred)

    f1 = f1_score(predictions, labels)
    accuracy_label_0 = calculate_accuracy_per_label(predictions, labels, label_value=0)
    accuracy_label_1 = calculate_accuracy_per_label(predictions, labels, label_value=1)

    return {"f1": f1, "accuracy_0": accuracy_label_0, "accuracy_1": accuracy_label_1}


def run_configs(
    configs: list[tuple[int, int]],
    file_path: str = "../impact_cite_test_results/run_1.csv",
):
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("Metric, F1, Accuracy_label_0, Accuracy_label_1\n")

    for config in configs:
        metrics = run(config[0], config[1])
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(
                f"{config}, {metrics['f1']}, {metrics['accuracy_0']}, {metrics['accuracy_1']}\n"
            )


if __name__ == "__main__":
    configs = [(90, 30), (150, 50), (300, 100), (500, 170)]
    run_configs(configs)