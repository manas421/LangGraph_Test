import os
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(
    api_key = os.getenv("OPENROUTER_API_KEY"),
    base_url = 'https://openrouter.ai/api/v1/',
    model = os.getenv("MODEL_NAME")
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(start_key=START, end_key= "chatbot")
graph_builder.add_edge(start_key="chatbot", end_key= END)

graph = graph_builder.compile()

user_input = input("Enter a message: ")
state = graph.invoke({"messages": [{"role":"user", "content":user_input}]})

print(state["messages"][-1].content)