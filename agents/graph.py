from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import END, StateGraph, add_messages, START
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool

from langchain_core.runnables import Runnable
from langgraph.types import Command

import os

from agents.prompts.prompt_manager import PromptManager

class AgentState(BaseModel):
    messages: Annotated[List, add_messages] = Field(default_factory=list)
    next_node: str | None = None

class LLMFactory:
    def __init__(self):
        self._llm = None

    def get_base_llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model="llama3.2:3b",
                temperature=0,
            )

        return self._llm
    
    def get_tool_llm(self, tools: List[BaseTool] = []):
        llm = self.get_base_llm()
        return llm.bind_tools(tools)
    
    def get_llm_with_structured_output(self, schema: dict | type):
        llm = self.get_base_llm()
        return llm.with_structured_output(schema)

class ToolAssistantNode:
    def __init__(self, tools):
        self.tools = tools
        llm_factory = LLMFactory()
        prompt_manager = PromptManager()

        self.llm = llm_factory.get_tool_llm(self.tools)

        tools_json = [
            tool.model_dump_json(include=["name", "description"])
            for tool in self.tools
        ]
        tools_context = "\n".join(tools_json)

        system_prompt = prompt_manager.get(
            "system_prompt",
            tools=tools_context
        )

        self.system_message = SystemMessage(content=system_prompt)


    def __call__(self, state):
        response = self.llm.invoke(
            [self.system_message] + state.messages
        )

        return {"messages": [response]}
    
class IntentClassifier(BaseModel):
    """Structured output for intent classification"""
    intent: Literal["tools", "default"] = Field(description="Next step in query processing")

class IntentClassifierNode:
    def __init__(self):
        llm_factory = LLMFactory()
        self.prompt_manager = PromptManager()
        self.llm = llm_factory.get_llm_with_structured_output(IntentClassifier)

    def __call__(self, state):

        system_prompt = self.prompt_manager.get("intent_classifier_prompt", message = state.messages[-1].content)
        self.system_message = SystemMessage(content = system_prompt)

        response = self.llm.invoke(
            [self.system_message]
        )

        # Dynamic routing based on intent
        if response.intent == "tools":
            next_node = "ToolAssistant_Node"
        elif response.intent == "default":
            next_node = "DefaultAssistant_Node"
        else:
            next_node = "DefaultAssistant_Node"

        return Command(
            update={
                "next_node": next_node
            },
            goto=next_node,
        )
    
class DefaultAssistantNode:
    def __init__(self):
        llm_factory = LLMFactory()
        prompt_manager = PromptManager()

        self.llm = llm_factory.get_base_llm()

        system_prompt = prompt_manager.get("default_prompt")
        self.system_message = SystemMessage(content=system_prompt)

    def __call__(self, state):
        response = self.llm.invoke(
            [self.system_message] + state.messages
        )

        return {"messages": [response]}
    
class AgentWorkflow:
    def __init__(self, tools: List[BaseTool] = []):
        self.tools = tools
        self.tool_assistant_node = ToolAssistantNode(tools)
        self.classify_intent_node = IntentClassifierNode()
        self.default_assistant_node = DefaultAssistantNode()
        
    async def _create_graph(self):
        """Create and configure the state graph for handling queries"""
        workflow = StateGraph(AgentState)

        workflow.add_node("ToolAssistant_Node", self.tool_assistant_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("IntentClassifier_Node", self.classify_intent_node)
        workflow.add_node("DefaultAssistant_Node", self.default_assistant_node)

        workflow.add_edge(START, "IntentClassifier_Node")
        workflow.add_conditional_edges(
            "ToolAssistant_Node",
            tools_condition,
        )
        workflow.add_edge("tools", "ToolAssistant_Node")
        workflow.add_edge("DefaultAssistant_Node", END)

        return workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    graph = AgentWorkflow([])._create_graph()