from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, field_validator
from langchain_fireworks import ChatFireworks
from json import JSONDecoder

# Define a lang graph state schema with just one parameter of employee_presence
class EmployeeSwipeState(BaseModel):
    morning_message: str

    @field_validator('morning_message')
    @classmethod
    def validate_mm(cls, value):
        # Check if value contains word "present"
        if "present" not in value:
            raise ValueError("Morning message should indicate whether you are present or not")
        return value

class EmployeePresence(BaseModel):
    employee_name: str
    employee_presence: bool

llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x22b-instruct")

# Define a greeting node that takes in a state and returns a new state with a greeting message
def greeting(state: EmployeeSwipeState)-> EmployeePresence:
    sys_content = """
    You will be provided a message. You should extract name and presence_status from the message respond in JSON format only.
    Your responses must adhere to the following structure:
    {
        "name": "John",
        "presence_status": "present"
    }
        Always ensure your entire response is enclosed in a valid JSON object. Do not include any text outside of the JSON structure.
    """
    sys_msg = SystemMessage(content=sys_content)
    #llm_response = llm.invoke([HumanMessage(content=state.morning_message + " Can you tell my name and my presence_status by responding only in JSON structure only?")])
    llm_response = llm.invoke( [sys_msg] + [HumanMessage(content=state.morning_message)])
    print(llm_response)
    decoder = JSONDecoder()
    llm_response_json = decoder.decode(s=llm_response.content)
    print(llm_response_json)
    if llm_response_json["presence_status"] == "present":
        return EmployeePresence(employee_name=llm_response_json["name"], employee_presence=True)
    else:
        return EmployeePresence(employee_name=llm_response_json["name"], employee_presence=False)

# Define a node that ask question about the presence of an employee
def ask_presence(state: EmployeePresence) -> MessagesState:
    if state.employee_presence:
        return {"messages": [SystemMessage(content=state.employee_name + " is present today.")]}
    else:
        return {"messages": [SystemMessage(content= state.employee_name + "has not reported today")]}

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
    lambda state: "send_bonus" if state.employee_presence else "send_message"
)
builder.add_edge("send_bonus", END)
builder.add_edge("send_message", END)

# Compile the graph
graph = builder.compile()
