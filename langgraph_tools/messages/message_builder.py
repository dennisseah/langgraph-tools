from typing import Literal

from langchain_core.messages import HumanMessage

type MessageKind = Literal["word_count", "summarize", "extract_entities"]

mapping = {
    "word_count": "count the number of words",
    "summarize": "summarize using summarization tool",
    "extract_entities": "get the entities using Azure service",
}


class MessageBuilder:
    def build(self, kinds: set[MessageKind], text: str) -> HumanMessage:
        actions = [mapping[kind] for kind in kinds]
        actions.sort()
        text_actions = "Perform the following actions: " if len(actions) > 1 else ""

        content = (
            f"{text_actions}" + ", ".join(actions) + f' on this blob of text. "{text}"'
        )
        return HumanMessage(content=content)
