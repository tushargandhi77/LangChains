from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate([
    ('system','You area helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []
# load chat history
with open('chat_history.txt') as file:
    chat_history.extend(file.readlines())


# create prompt template
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'What is the status of my order?'})

print(prompt)