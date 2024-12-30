import argparse
import asyncio
import os
import sys

from langgraph_tools.hosting import container
from langgraph_tools.messages.message_builder import MessageKind
from langgraph_tools.protocols.i_azure_openai_service import IAzureOpenAIService
from langgraph_tools.tools import count_words, get_entities, summarize


class ExecutorBase:
    def __init__(self):
        self.tools = [count_words, get_entities, summarize]
        self.llm_model = container[IAzureOpenAIService].get_model()
        self.llm_model_with_tools = self.llm_model.bind_tools(self.tools)

    def parse_args(self) -> tuple[str, list[MessageKind]]:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--file", help="file to process", required=True)
        parser.add_argument(
            "-c", "--count-words", help="count words", action="store_true"
        )
        parser.add_argument("-s", "--summarize", help="summarize", action="store_true")
        parser.add_argument(
            "-e", "--extract-entities", help="summarize", action="store_true"
        )
        args = parser.parse_args()

        actions = []
        if args.count_words:
            actions.append("word_count")
        if args.summarize:
            actions.append("summarize")
        if args.extract_entities:
            actions.append("extract_entities")

        if not actions:
            print("At least one action must be selected")
            sys.exit(1)

        return args.file, actions

    async def _invoke(self, text: str, actions: list[MessageKind]) -> str: ...

    def invoke(self):
        file, actions = self.parse_args()

        path = os.path.join("data", file)
        with open(path, "r") as fp:
            text = fp.read()
            result = asyncio.run(self._invoke(text, actions))
            print(result)
            sys.exit(0)
