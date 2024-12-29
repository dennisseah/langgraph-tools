from attr import dataclass
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langgraph_tools.llm_models.openai_model import get_model
from langgraph_tools.protocols.i_summarization_service import ISummarizationService


@dataclass
class SummarizationService(ISummarizationService):
    def summarize(self, content: str) -> str:
        template = PromptTemplate(
            template="""Summarize the following text in English:

            `{text}`

            Do not provide anything other than the translation.""",
            input_variables=["text"],
        )

        llm_model = get_model()

        chain = template | llm_model | StrOutputParser()
        return chain.invoke({"text": content})
