from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

HuggingFaceEmbeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# text = "Delhi is the capital of India."

# vector = HuggingFaceEmbeddings.embed_query(text)

# print(str(vector))

document = [
    "Delhi is the capital of India.",
    "Delhi is the capital of India. It is a city with a rich history and diverse",
]

vectors = HuggingFaceEmbeddings.embed_documents(document)

print(str(vectors))
