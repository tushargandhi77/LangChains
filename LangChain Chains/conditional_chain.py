from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")

parser_pydantic = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following  text into positive, negative \n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser_pydantic.get_format_instructions()}
)


classifier_chain = prompt1 | model | parser_pydantic


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback'],
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback'],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive',prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative',prompt3 | model | parser),
    RunnableLambda(lambda x : "Could not find sentiment in the feedback")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a good phone'})

print(result)

chain.get_graph().print_ascii()