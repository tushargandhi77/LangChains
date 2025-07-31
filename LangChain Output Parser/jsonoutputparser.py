from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give ma the name, age and City of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({})

print(final_result)
print(type(final_result))