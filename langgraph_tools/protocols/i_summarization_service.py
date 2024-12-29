from typing import Protocol


class ISummarizationService(Protocol):
    def summarize(self, content: str) -> str: ...
