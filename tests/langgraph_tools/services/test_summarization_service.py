from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from langgraph_tools.services.summarization_service import SummarizationService


def test_summarization_service(mocker: MockerFixture):
    mocked_model = MagicMock()

    mocker.patch(
        "langgraph_tools.services.summarization_service.get_model",
        return_value=mocked_model,
    )

    mocker.patch(
        "langgraph_tools.services.summarization_service.StrOutputParser",
        return_value=MagicMock(),
    )

    service = SummarizationService()
    content = "The quick brown fox jumps over the lazy dog."
    result = service.summarize(content)
    assert result is not None
