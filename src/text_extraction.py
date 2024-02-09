import re


class TextExtraction:
    def __init__(
        self,
        article_dict: dict,
        previous_context_tokens: int = None,
        following_context_tokens: int = None,
        footnote_text: bool = True,
        footnote_mask: bool = True,
    ):
        self.previous_context_tokens = previous_context_tokens
        self.following_context_tokens = following_context_tokens
        self.footnote_text = footnote_text
        self.footnote_mask = footnote_mask

        self.article_text = article_dict["article"]
        # keep in mind keys of this dict are strings of integers
        self.footnote_dict = article_dict["footnotes"]

    def generate_context(self, footnote_number: int):
        # Find the index of the footnote in the article text
        footnote_index = self.article_text.find(f"[CITATION-{footnote_number}]")

        if footnote_index == -1:
            raise ValueError(f"Footnote {footnote_number} not found in article text")

        # Get the content of the specified footnote
        footnote_content = self.footnote_dict[str(footnote_number)]

        # Extract the relevant context based on options
        start_index = max(0, self.find_previous_token_index(footnote_index))
        end_index = min(
            len(self.article_text),
            self.find_following_token_index(
                footnote_index + len(f"[CITATION-{footnote_number}]")
            ),
        )

        context = self.article_text[start_index:end_index].strip()

        # Apply footnote mask if required
        if self.footnote_mask:
            context = self.replace_citations(context, footnote_number)

        # Add footnote text if required
        if self.footnote_text:
            context += "   \n   " + f"Footnote {footnote_number}: {footnote_content}"

        return context

    @staticmethod
    def replace_citations(text: str, footnote_number: int) -> str:
        citation_pattern = r"\[CITATION-(\d+)\]"

        def replacer(match):
            if match.group(1) == str(footnote_number):
                return match.group(0)
            else:
                return "[MASK]"

        replaced_text = re.sub(citation_pattern, replacer, text)
        return replaced_text

    def find_previous_token_index(self, index: int) -> int:
        if self.previous_context_tokens is None:
            return index

        count_tokens = 0
        while count_tokens < self.previous_context_tokens and index > 0:
            index -= 1
            if self.article_text[index].isspace():
                count_tokens += 1
        return index

    def find_following_token_index(self, index: int) -> int:
        if self.following_context_tokens is None:
            return index

        count_tokens = 0
        while count_tokens < self.following_context_tokens and index < len(
            self.article_text
        ):
            index += 1
            if self.article_text[index - 1].isspace():
                count_tokens += 1

        return index
