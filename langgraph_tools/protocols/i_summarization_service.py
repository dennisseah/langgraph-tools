from typing import Protocol


class ISummarizationService(Protocol):
    def summarize(self, content: str) -> str:
        """
        Summarize the content and return the summary.

        :param content: The content to summarize.
        :return: The summary of the content.
        """
        ...
