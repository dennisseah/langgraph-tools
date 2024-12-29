from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.ai.textanalytics._models import CategorizedEntity, RecognizeEntitiesResult
from pytest_mock import MockerFixture

from langgraph_tools.services.text_extraction_service import (
    TextExtractionService,
    TextExtractionServiceEnv,
)


@pytest.fixture
def mock_env() -> Callable[[bool], TextExtractionServiceEnv]:
    def wrapper(has_key: bool) -> TextExtractionServiceEnv:
        return TextExtractionServiceEnv(
            azure_text_analytics_endpoint="https://fake-endpoint.com",
            azure_text_analytics_key="fake-key" if has_key else None,
        )

    return wrapper


@pytest.fixture
def mock_service(mock_env) -> Callable[[bool], TextExtractionService]:
    def wrapper(has_key: bool) -> TextExtractionService:
        return TextExtractionService(env=mock_env(has_key), logger=MagicMock())

    return wrapper


def test_text_extraction_service_key(
    mock_service: Callable[[bool], TextExtractionService], mocker: MockerFixture
):
    mocker.patch("azure.ai.textanalytics.TextAnalyticsClient")
    assert mock_service(True).get_client() is not None


def test_text_extraction_service_without_key(
    mock_service: Callable[[bool], TextExtractionService], mocker: MockerFixture
):
    mocker.patch("azure.ai.textanalytics.TextAnalyticsClient")
    assert mock_service(False).get_client() is not None


@pytest.mark.asyncio
async def test_recognize_entities_no_content(
    mock_service: Callable[[bool], TextExtractionService],
):
    service = mock_service(True)
    assert await service.recognize_entities("") == []


@pytest.mark.asyncio
async def test_recognize_entities(
    mock_service: Callable[[bool], TextExtractionService],
):
    service = mock_service(True)
    mock_client = MagicMock()
    mock_client.recognize_entities = AsyncMock(
        return_value=[
            RecognizeEntitiesResult(
                id="0",
                entities=[
                    CategorizedEntity(
                        text="Studies",
                        category="Event",
                        subcategory="Event",
                        offset=0,
                        length=7,
                        confidence_score=0.99,
                    )
                ],
            )
        ]
    )
    service.get_client = MagicMock(return_value=mock_client)

    input = "Studies have shown that regular physical activity is associated with a longer lifespan, reducing the risk"  # noqa E501

    results = await service.recognize_entities(input)

    assert len(results) == 1
    assert results[0].id == "0"
    assert results[0].entities[0].text == "Studies"
    assert results[0].entities[0].sentence == input
