from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

result = model.invoke("what is the capital of India?")

print(result.content)

# The capital of India is **New Delhi**.