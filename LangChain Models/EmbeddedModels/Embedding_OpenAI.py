from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)

embeddings = OpenAIEmbeddings.embed_query("Hello world")

print(embeddings)