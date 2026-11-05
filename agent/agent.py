from typing import Annotated
from dotenv import load_dotenv
import os
from headroom.integrations.langchain import create_compress_tool_messages_node
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode

from agent.config.state import State
from agent.tools.sandbox import sandbox

load_dotenv()

def llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("api key is not set")
    return ChatOpenAI(
        model = 'deepseek-v4-pro',
        api_key = api_key,
        base_url = 'https://api.deepseek.com',
    )

class Agent:
    def __init__(self):
        self.agent =None
        self.llm=None
        self.tools=None
        self.memory=None
        self.graph=None
    
    async def setup(self):
        self.tools=[sandbox]
        self.llm=llm()
        self.agent = self.llm.bind_tools(self.tools)
        self.graph = None
        self.memory = None
    
    def worker(self,state:State):
        system_message = """ """

        found_system_message = False
        messages = state["messages"]

        for message in messages:
            if isinstance(message,SystemMessage):
                messages = [
                    SystemMessage(content = system_message),
                    *[m for m in state["messages"] if not isinstance(m,SystemMessage)]
                ]
                found_system_message = True
        if not found_system_message:
            messages = [SystemMessage(content = system_message)]+ messages

        response = self.agent.invoke(messages)
        return {
            "messages": [response]
        }

    def agent_router(self,state:State):
        last_message = state["messages"][-1]
        if isinstance(last_message,AIMessage) and last_message.tool_calls:
            return "tools"   
        return END  
                                                                          
    async def build_graph(self):
        graph_builder = StateGraph(State)

        #Nodes
        graph_builder.add_node("agent",self.worker)
        graph_builder.add_node("tools",ToolNode(self.tools))
        graph_builder.add_node("compress",create_compress_tool_messages_node(min_tokens_to_compress=500))

        #Edges
        graph_builder.add_edge(START,"agent")
        graph_builder.add_conditional_edges(
            "agent",
            self.agent_router,
            {
                "tools":"tools",
                END:END
            }
        )
        graph_builder.add_edge("tools","compress")
        graph_builder.add_edge("compress","agent")
      

        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self,state:State):
        config = {"configurable": {"thread_id": "conversation_1"}}
        result = await self.graph.ainvoke(state,config=config)
        return result 