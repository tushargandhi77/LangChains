from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()


# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]


# Initialize the model

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# create a vector store

vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)


# crate retriever

# 1
similarity_retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={"k":5})

# 2
multiquery_retriever = MultiQueryRetriever.from_llm(
   retriever=vector_store.as_retriever(search_kwargs={"k":5}), # you can also take mmr
   llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
)

# query

query = "How to improve energy levels and maintain balance"

# retriever result

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)



# --- Result 1 ---
# Drinking sufficient water throughout the day helps maintain metabolism and energy.

# --- Result 2 ---
# --- Result 2 ---
# Mindfulness and controlled breathing lower cortisol and improve mental clarity.

# --- Result 3 ---
# Consuming leafy greens and fruits helps detox the body and improve longevity.

# --- Result 4 ---
# Deep sleep is crucial for cellular repair and emotional regulation.

# --- Result 5 ---
# Regular walking boosts heart health and can reduce symptoms of depression.
# ******************************************************************************************************************************************************

# --- Result 1 ---
# Drinking sufficient water throughout the day helps maintain metabolism and energy.

# --- Result 2 ---
# Mindfulness and controlled breathing lower cortisol and improve mental clarity.

# --- Result 3 ---
# Consuming leafy greens and fruits helps detox the body and improve longevity.

# --- Result 4 ---
# Regular walking boosts heart health and can reduce symptoms of depression.

# --- Result 5 ---
# Deep sleep is crucial for cellular repair and emotional regulation.

