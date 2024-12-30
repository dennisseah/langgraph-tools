from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from langgraph_tools.executors.executor_base import ExecutorBase
from langgraph_tools.messages.message_builder import MessageBuilder, MessageKind


class ChainExecutor(ExecutorBase):
    async def _invoke(self, text: str, actions: list[MessageKind]) -> str:
        message = MessageBuilder().build(set(actions), text)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{text}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(self.llm_model, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        result = await agent_executor.ainvoke({"text": message})
        return result["output"]


if __name__ == "__main__":
    ChainExecutor().invoke()
