from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "Explain the following  joke {text}",
    input_variables=["text"],
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke({"topic":"AI"})

print(result)
