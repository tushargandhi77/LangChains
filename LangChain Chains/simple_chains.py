from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Generate a 5 interesting facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

# chains

chain = prompt | model | parser

result = chain.invoke({"topic":"India"})

print(result)

chain.get_graph().print_ascii()  # Print the chain graph in ASCII format
