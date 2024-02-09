import os.path
from .utils import get_completion_from_messages
from .text_extraction import TextExtraction
import json
import pandas as pd
from tqdm import tqdm


def create_opinionated(prompt: str) -> str:
    system_message = """
    You are an advanced AI writer. As an input, you will get a source input, which contains parts of a source text, which is aligned to a certain citation. Your task is to CHANGE the source input such that it has an opinion towards or against the citation.  You are only allowed to add ONE or TWO sentences which should change the source input to an opinionated sentence. 

    You are NOT ALLOWED to make any changes to the source input other than adding the sentences.

    The citation is marked through [CITATION-x] where x stands for the citation number. 

    Also, the source input will contain the citation itself featuring the author's name and title of the cited paper(s). To better connect the opinion to the right citation.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    prediction = get_completion_from_messages(
        messages, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0
    )
    return prediction


if __name__ == "__main__":
    # These parameters were used to generate
    # "synthetic_data/A_Colonial_Celebrity_in_the_New_Attention_Economy_Cecil_Rhodess_Cape-to-Cairo_Telegraph_and_Railway_Negotiations_in_1899.csv"

    filename = "A_Colonial_Celebrity_in_the_New_Attention_Economy_Cecil_Rhodess_Cape-to-Cairo_Telegraph_and_Railway_Negotiations_in_1899.json"
    filepath = "./all_data_articles/"
    synthetic_filepath = "./synthetic_data/"

    def generate_synthetic_data(filepath: str, filename: str, synthetic_filepath: str):

        with open(os.path.join(filepath, filename), "r", encoding="utf-8") as file:
            article_dict = json.load(file)

        previous_context_tokens = 70
        following_context_tokens = 30

        footnotes = article_dict["footnotes"]

        text_extractor = TextExtraction(
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
        )

        generated_contexts = []

        for footnote_number in tqdm(footnotes):
            context = text_extractor.generate_context(int(footnote_number))

            context_and_footnote = (
                context + "\n\n" + "CITATION:\n" + footnotes[footnote_number]
            )

            generated_context = create_opinionated(context_and_footnote)
            generated_contexts.append(generated_context)

        generated_contexts_df = pd.DataFrame(
            columns=["footnote_number", "generated_context", "footnote_text", "label"]
        )

        generated_contexts_df["footnote_number"] = footnotes.keys()
        generated_contexts_df["generated_context"] = generated_contexts
        generated_contexts_df["footnote_text"] = footnotes.values()
        generated_contexts_df["label"] = [1] * len(footnotes)

        generated_contexts_df.to_csv(
            os.path.join(synthetic_filepath, filename.replace(".json", ".csv")),
            index=False,
        )
