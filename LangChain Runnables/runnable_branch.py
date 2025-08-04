from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


prompt1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template = "summarize the following report: {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

report_generator_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_generator_chain, branch_chain)

result = final_chain.invoke({"topic":"Russia vs Ukraine"})

print(result)
