from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# Define a lang graph state schema with just one parameter of employee_presence
class EmployeeSwipeState(TypedDict):
    morning_message: str

class EmployeePresence(TypedDict):
    employee_name: str
    employee_presence: bool

# Define a greeting node that takes in a state and returns a new state with a greeting message
def greeting(state: EmployeeSwipeState)-> EmployeePresence:
    if state["morning_message"] == "Good morning, I am present today":
        return {"employee_name": "John", "employee_presence": True}
    else :
        return {"employee_name": "John", "employee_presence": False}

# Define a node that ask question about the presence of an employee
def ask_presence(state: EmployeePresence) -> MessagesState:
    if state["employee_presence"]:
        return {"messages": [SystemMessage(content="Employee is present today.")]}
    else:
        return {"messages": [SystemMessage(content="Employee is not reported today")]}

# Define a node that takes in a state and sends bonus if the employee is present
def send_bonus(state: MessagesState):
    return {"messages": [SystemMessage(content="Bonus sent!")]}

# Define a node that takes in a state and sends a message if the employee is not present
def send_message(state: MessagesState):
    return {"messages": [SystemMessage(content="The employee is not present.")]}

# Define state graph builder with above nodes and conditional edges
builder = StateGraph(EmployeeSwipeState)
builder.add_node("greeting", greeting)
builder.add_node("ask_presence", ask_presence)
builder.add_node("send_bonus", send_bonus)
builder.add_node("send_message", send_message)
builder.add_edge(START, "greeting")
builder.add_edge("greeting", "ask_presence")
builder.add_conditional_edges(
    "ask_presence",
    lambda state: "send_bonus" if state["employee_presence"] else "send_message"
)
builder.add_edge("send_bonus", END)
builder.add_edge("send_message", END)

# Compile the graph
graph = builder.compile()
