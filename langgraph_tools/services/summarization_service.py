from attr import dataclass
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langgraph_tools.protocols.i_azure_openai_service import IAzureOpenAIService
from langgraph_tools.protocols.i_summarization_service import ISummarizationService


@dataclass
class SummarizationService(ISummarizationService):
    openai_service: IAzureOpenAIService

    def summarize(self, content: str) -> str:
        template = PromptTemplate(
            template="""Summarize the following text:

            `{text}`

            Do not provide anything other than the translation.""",
            input_variables=["text"],
        )

        chain = template | self.openai_service.get_model() | StrOutputParser()
        return chain.invoke({"text": content})
