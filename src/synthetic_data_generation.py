from utils import get_completion_from_messages


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
    prediction = get_completion_from_messages(messages, max_tokens=256, top_p=1, frequency_penalty=0,
                                              presence_penalty=0)
    return prediction
