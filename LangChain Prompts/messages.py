from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain")
]

result = model.invoke(messages)

messages = AIMessage(content=result.content)

print(messages)


