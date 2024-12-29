import asyncio
import logging

from langchain_core.tools import tool

from langgraph_tools.hosting import container
from langgraph_tools.protocols.i_summarization_service import ISummarizationService
from langgraph_tools.protocols.i_text_extraction_service import ITextExtractionService

logger = container[logging.Logger]


@tool
def count_words(text: str) -> int:
    """Count the number of words in a text"""
    logger.info("[TOOL]: Counting words")
    return len(text.split())


@tool
def get_entities(text: str) -> list[str]:
    """Extract entities from a text using Azure service"""
    logger.info("[TOOL]: Extracting entities")
    result = asyncio.run(container[ITextExtractionService].recognize_entities(text))
    return [r.text for r in result[0].entities]


@tool
def summarize(text: str) -> str:
    """Summarize a text"""
    logger.info("[TOOL]: Summarizing")
    return container[ISummarizationService].summarize(text)
