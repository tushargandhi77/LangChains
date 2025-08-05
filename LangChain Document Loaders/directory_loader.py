from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path ='LangChain Document Loaders/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)
# print(docs[0].metadata)

# for doc in docs:
#     print(doc.metadata)


# lazy load

docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)