from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


embeddings = GoogleGenerativeAIEmbeddings.embed_query("Hello world")

print(embeddings[:10])