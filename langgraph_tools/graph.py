import argparse
import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from langgraph_tools.llm_models.openai_model import get_model
from langgraph_tools.tools import count_words, get_entities, summarize

# tools, count_words, get_entities and summarize
tools = [count_words, get_entities, summarize]

# bind tools to the model
llm_model_with_tools = get_model().bind_tools(tools)


def should_continue(state: MessagesState):
    messages = state["messages"]
    return "tools" if messages[-1].tool_calls else END  # type: ignore


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


async def run(text: str) -> str:
    app = create_graph()
    messages: list[HumanMessage | AIMessage | ToolMessage] = []
    async for value in app.astream(
        {
            "messages": [
                HumanMessage(
                    content="Count the number of words, summarize "
                    f'and get the entities in this blob of text, "{text}".',
                )
            ]
        },
        stream_mode="values",
    ):
        messages = value["messages"]

    return str(messages[-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="file to process", required=True)

    path = os.path.join("data", parser.parse_args().file)
    with open(path, "r") as file:
        text = file.read()
        result = asyncio.run(run(text))
        print(result)
