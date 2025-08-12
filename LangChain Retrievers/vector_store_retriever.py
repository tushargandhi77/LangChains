from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# step 1 

documents =[
    Document(page_content="LangChain helps developers build llm applications easily."),
    Document(page_content="Chroma is a vector database optimized for llm-based search."),
    Document(page_content="Embeddings  convert text to high dimensional vectors."),
    Document(page_content="Gemini Provides powerful embedding models.")
]

# step 2
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# step 3
vector_store = Chroma.from_documents(
    documents=documents,
    embedding = embedding_model,
    collection_name='my_collection'
)

# step 4
retriever = vector_store.as_retriever(search_kwargs={"k":2})

query = "what is chroma used for"
result = retriever.invoke(query)

for i,doc in enumerate(result):
    print(f"\n---Result {i+1} ---")
    print(f"content:\n{doc.page_content} ....")