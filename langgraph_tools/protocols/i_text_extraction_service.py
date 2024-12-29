from typing import Protocol

from langgraph_tools.models.recognized_entities import (
    RecognizedEntities,
)


class ITextExtractionService(Protocol):
    async def recognize_entities(self, content: str) -> list[RecognizedEntities]: ...
