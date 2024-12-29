from typing import Protocol

from langgraph_tools.models.recognized_entities import (
    RecognizedEntities,
)


class ITextExtractionService(Protocol):
    async def recognize_entities(self, content: str) -> list[RecognizedEntities]:
        """
        Recognize entities in the content and return the recognized entities.

        :param content: The content to recognize entities in.
        :return: The recognized entities in the content.
        """
        ...
