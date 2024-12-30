from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from langgraph_tools.executors.executor_base import ExecutorBase
from langgraph_tools.messages.message_builder import MessageBuilder, MessageKind


class GraphExecutor(ExecutorBase):
    def should_continue(self, state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        return (
            "tools"
            if isinstance(last_message, AIMessage) and last_message.tool_calls
            else END
        )

    def call_model(self, state: MessagesState):
        messages = state["messages"]
        response = self.llm_model_with_tools.invoke(messages)
        return {"messages": [response]}

    def create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(MessagesState)

        graph.add_node("agent", self.call_model)
        tool_node = ToolNode(self.tools)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self.should_continue, ["tools", END])
        graph.add_edge("tools", "agent")

        return graph.compile()

    async def _invoke(self, text: str, actions: list[MessageKind]) -> str:
        app = self.create_graph()
        last_message: BaseMessage | None = None
        message = HumanMessage(content=MessageBuilder().build(set(actions), text))

        async for value in app.astream(
            {"messages": [message]},
            stream_mode="values",
        ):
            last_message = value["messages"][-1]

        return str(last_message.content) if last_message else "not results"


if __name__ == "__main__":
    GraphExecutor().invoke()
