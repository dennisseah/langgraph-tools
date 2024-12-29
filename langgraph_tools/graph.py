import argparse
import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from langgraph_tools.llm_models.openai_model import get_model
from langgraph_tools.tools import count_words, get_entities

# two tools, count_words and get_entities
tools = [count_words, get_entities]

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


async def run(text: str):
    app = create_graph()
    messages: list[HumanMessage | AIMessage | ToolMessage] = []
    async for value in app.astream(
        {
            "messages": [
                HumanMessage(
                    content=f"""Count the number of words and get the entities in this blob of text, "{text}".""",  # noqa E501
                )
            ]
        },
        stream_mode="values",
    ):
        messages = value["messages"]

    print(messages[-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="file to process", required=True)

    path = os.path.join("data", parser.parse_args().file)
    with open(path, "r") as file:
        text = file.read()
        asyncio.run(run(text))
