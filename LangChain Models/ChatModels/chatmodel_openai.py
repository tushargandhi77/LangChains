from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=0.3,max_completion_tokens=1034) # temperature controls randomness and creativity

# tokens example as words

result = model.invoke("what is the capital of France?")

print(result.content)