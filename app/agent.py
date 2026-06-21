from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langsmith import traceable

from app.prompts import SYSTEM_MESSAGE
from app.tools import create_retriever_tool
from app.config import get_settings
from app.monitoring import get_logger

logger = get_logger()

# === Agent State ===

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str

class ProductionAgent:
    """
    LangGraph Agent with:
    - Retry on failure (model fallback)
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()
        tools = [create_retriever_tool()]
        self._tools_dict = {t.name: t for t in tools}
        self.primary_llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0, # we handle retries ourselves
            api_key=settings.openai_api_key
        ).bind_tools(tools=tools)

        self.fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key
        ).bind_tools(tools=tools)

        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph State machine"""

        def process_message(state: AgentState) -> dict:
            """Try to process the message with the primary model."""
            try:
                system_message = SystemMessage(content=SYSTEM_MESSAGE)
                response = self.primary_llm.invoke([system_message] + state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary"
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": ""
                }
            
        def try_fallback(state: AgentState) -> dict:
            try:
                system_message = SystemMessage(content=SYSTEM_MESSAGE)
                response = self.primary_llm.invoke([system_message] + state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback"
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": ""
                }
        
        def handle_error(state: AgentState) -> dict:
            """ Return a graceful error message."""
            return {
                "messages": [
                    AIMessage(content="I am sorry, I am having a trouble processing your request right now. Please try again in a moment.")
                ],
                "model_used": "error_handler"
            }
        
        async def call_retriever_agent(state: AgentState) -> dict:
            """Execute tool calls from the LLM's response."""
            tool_calls = state["messages"][-1].tool_calls
            results = []
            for t in tool_calls:
                logger.info(f"Calling tool: {t['name']} with query: {t['args'].get('query', 'No query provided.')}")
                if not t['name'] in self._tools_dict:
                    logger.warning(f"\nTool {t['name']} does not exist.")
                    result = "Incorrect Tool Name, Please Retry and Select tool from list of available tools"
                else:
                    result = await self._tools_dict[t['name']].ainvoke(t['args'].get('query', ''))
                    logger.info(f"Result length: {len(str(result))}")

                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
                logger.info("Tools execution complete. Back to model!")
                return {"messages": results}
        
        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt."""
            if state.get("error") is None:
                last_message = state["messages"][-1]
                if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
                    return "tool_call"
                else:
                    return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"
            
        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback model attempt."""
            if state.get("error") is None:
                last_message = state["messages"][-1]
                if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
                    return "tool_call"
                else:
                    return "done"
            else:
                return "error"
            
        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("retriever_agent", call_retriever_agent)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {
                "done": END,
                "tool_call": "retriever_agent",
                "fallback": "fallback",
                "error": "error"
            }
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {
                "done": END,
                "tool_call": "retriever_agent",
                "error": "error"
            }
        )
        graph.add_edge("retriever_agent", "process")
        graph.add_edge("error", END)

        return graph.compile()
    
    @traceable(name="production_agent_invoke")
    async def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """

        result = await self.graph.ainvoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": ""
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error")
        }
