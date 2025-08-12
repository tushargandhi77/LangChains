from langchain_community.retrievers import WikipediaRetriever

# initialize the retriever
retriever = WikipediaRetriever(top_k_results=2,lang='en')

# query
query = "The geopolitical history of india and pakistan from the perspective of a chinese"


# get relevant documents
docs = retriever.invoke(query)

for i,doc in enumerate(docs):
    print(f"\n---Result {i+1} ---")
    print(f"content:\n{doc.page_content} ....") 

