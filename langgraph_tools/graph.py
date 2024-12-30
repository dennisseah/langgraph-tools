import argparse
import asyncio
import os
import sys

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from langgraph_tools.hosting import container
from langgraph_tools.messages.message_builder import MessageBuilder, MessageKind
from langgraph_tools.protocols.i_azure_openai_service import IAzureOpenAIService
from langgraph_tools.tools import count_words, get_entities, summarize

# tools, count_words, get_entities and summarize
tools = [count_words, get_entities, summarize]

# bind tools to the model
llm_model_with_tools = container[IAzureOpenAIService].get_model().bind_tools(tools)


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    return (
        "tools"
        if isinstance(last_message, AIMessage) and last_message.tool_calls
        else END
    )


def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_model_with_tools.invoke(messages)
    return {"messages": [response]}


def create_graph() -> CompiledStateGraph:
    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_model)
    tool_node = ToolNode(tools)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


async def invoke(text: str, actions: list[MessageKind]) -> str:
    app = create_graph()
    last_message: BaseMessage | None = None
    message = HumanMessage(content=MessageBuilder().build(set(actions), text))

    async for value in app.astream(
        {"messages": [message]},
        stream_mode="values",
    ):
        last_message = value["messages"][-1]

    return str(last_message.content) if last_message else "not results"


def parse_args() -> tuple[str, list[MessageKind]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="file to process", required=True)
    parser.add_argument("-c", "--count-words", help="count words", action="store_true")
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


if __name__ == "__main__":
    file, actions = parse_args()

    path = os.path.join("data", file)
    with open(path, "r") as fp:
        text = fp.read()
        result = asyncio.run(invoke(text, actions))
        print(result)
        sys.exit(0)
