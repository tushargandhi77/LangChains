from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough

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

joke_generator_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explanation': RunnableSequence(prompt2,model,parser)
    }
)


final_chain = RunnableSequence(joke_generator_chain,parallel_chain)


result = final_chain.invoke({"topic":"Cricket"})

print(result)

