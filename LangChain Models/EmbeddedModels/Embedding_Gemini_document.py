from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

document = [
    "Hello world",
    "This is a test document for embedding.",
]

embeddings = GoogleGenerativeAIEmbeddings.embed_documents(document)

print(embeddings[0][:10])  # Print first 10 dimensions of the first embedding
print(embeddings[1][:10])  # Print first 10 dimensions of the second