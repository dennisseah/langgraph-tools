import os
from unittest import mock

from pytest_mock import MockerFixture

from langgraph_tools.llm_models.openai_model import get_model

envvars = {
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    "AZURE_OPENAI_API_VERSION": "2021-07-01",
    "AZURE_OPENAI_DEPLOYED_MODEL_NAME": "test_model",
}

envvars_with_key = {**envvars, **{"AZURE_OPENAI_API_KEY": "test_key"}}


@mock.patch.dict(os.environ, envvars_with_key, clear=True)
def test_with_api_key(mocker: MockerFixture):
    mock_openai = mocker.patch(
        "langgraph_tools.llm_models.openai_model.AzureChatOpenAI"
    )
    get_model()
    mock_openai.assert_called_once_with(
        api_key=envvars_with_key["AZURE_OPENAI_API_KEY"],
        api_version=envvars_with_key["AZURE_OPENAI_API_VERSION"],
        model=envvars_with_key["AZURE_OPENAI_DEPLOYED_MODEL_NAME"],
        temperature=0.5,
    )


@mock.patch.dict(os.environ, envvars, clear=True)
def test_without_api_key(mocker: MockerFixture):
    mock_openai = mocker.patch(
        "langgraph_tools.llm_models.openai_model.AzureChatOpenAI"
    )

    mock_credential = mock.MagicMock()
    mock_credential.get_token = mock.MagicMock(
        return_value=mock.MagicMock(token="test_token")
    )

    mocker.patch(
        "langgraph_tools.llm_models.openai_model.DefaultAzureCredential",
        return_value=mock_credential,
    )
    get_model()
    mock_openai.assert_called_once_with(
        azure_ad_token="test_token",
        api_version=envvars["AZURE_OPENAI_API_VERSION"],
        model=envvars["AZURE_OPENAI_DEPLOYED_MODEL_NAME"],
        temperature=0.5,
    )
