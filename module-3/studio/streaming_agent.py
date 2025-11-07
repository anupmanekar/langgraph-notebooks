import os
from dotenv import load_dotenv
import asyncio
from langchain_fireworks import ChatFireworks
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

load_dotenv()

separator = "---------------------------------------------------------------------------------------------------------------------------------"

# LLM
model = ChatFireworks(model="accounts/fireworks/models/mixtral-8x22b-instruct", api_key=os.environ.get("FIREWORKS_API_KEY")) 

# State 
class State(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state: State, config: RunnableConfig):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages, config)
    return {"messages": response}

def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

config = { "thread_id": 123}

for chunk in graph.stream({"messages": [HumanMessage("Hi there, I am Jane John from NY")]}, config=config, stream_mode="values"):
    print(chunk)

print(separator)
config = { "thread_id": 55}

for chunk in graph.stream({"messages": [HumanMessage("Hi there, I am Jane John from NY")]}, config=config, stream_mode="updates"):
    print(chunk)

print(separator)

config = { "thread_id": 77}

async def main(config):
    async for event in graph.astream_events({"messages": [HumanMessage("Hi there, I am Jane John from NY")]}, config=config, version="v2"):
        if (event["event"] == "on_chat_model_stream"):
            data = event["data"]
            print(f"{data['chunk'].content} |")

asyncio.run(main(config))

