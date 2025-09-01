from langchain_core.tools import tool

# step 1 - create a function

def multiply(a,b):
    """ Multiply two numbers"""
    return a * b

# step 2 -- add type hints

@tool
def multiply(a: int,b: int) -> int:
    """ Multiply two numbers"""
    return a * b

result = multiply.invoke({"a":5,"b":6})

print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)


print(multiply.args_schema.model_json_schema())