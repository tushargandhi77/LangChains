from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Bumrah"

documents_embeddings = embeddings.embed_documents(documents) # five vectors
query_embedding = embeddings.embed_query(query) # one vector

scores = cosine_similarity([query_embedding],documents_embeddings)[0] # 2 D list always

max_scores = sorted(list(enumerate(scores)),key=lambda x: x[1],reverse=True)

max_scores_indices = max_scores[0][0]

print(f"similarity score: {max_scores[0][1]} for query: {query} and document: {documents[max_scores_indices]}")