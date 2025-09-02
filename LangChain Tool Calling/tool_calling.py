from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()

# tool creation
@tool
def multiply(a: int,b: int)->int:
    """Given 2 numbers a and b this tool return their product"""
    return a * b

# model creation
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# tool binding
llm_with_tool = model.bind_tools([multiply])


# tool calling 
result =  llm_with_tool.invoke("Can you multiply 2 and 3?")
# {'name': 'multiply', 'args': {'a': 2.0, 'b': 3.0}, 'id': '39d44519-ee76-4dfe-af0f-2f665b2f1dfa', 'type': 'tool_call'}
print(result.tool_calls[0]['args'])

#tool Execution
# print(multiply.invoke(result.tool_calls[0]['args']))
print(multiply.invoke(result.tool_calls[0]))




