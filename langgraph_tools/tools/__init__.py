import asyncio

from langchain_core.tools import tool

from langgraph_tools.hosting import container
from langgraph_tools.protocols.i_text_extraction_service import ITextExtractionService


@tool
def count_words(text: str) -> int:
    """Count the number of words in a text"""
    return len(text.split())


@tool
def get_entities(text: str) -> list[str]:
    """Extract entities from a text"""
    result = asyncio.run(container[ITextExtractionService].recognize_entities(text))
    return [r.text for r in result[0].entities]
