import os

from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI


def get_model(temperature: float = 0.5):
    if "AZURE_OPENAI_API_KEY" in os.environ:
        return AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],  # type: ignore
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],  # type: ignore
            model=os.environ["AZURE_OPENAI_DEPLOYED_MODEL_NAME"],
            temperature=temperature,
        )

    return AzureChatOpenAI(
        azure_ad_token=DefaultAzureCredential()
        .get_token("https://cognitiveservices.azure.com/.default")
        .token,  # type: ignore
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],  # type: ignore
        model=os.environ["AZURE_OPENAI_DEPLOYED_MODEL_NAME"],
        temperature=temperature,
    )
