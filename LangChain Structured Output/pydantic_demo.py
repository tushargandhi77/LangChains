from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Tushar"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10,default=5.0,description="CGPA must be between 0 and 10")


# new_student = Student(name="John Doe")
# print(new_student)

# new_student = {"name": 32}  # Input should be a valid string [type=string_type, input_value=32, input_type=int]
#     For further information visit https://errors.pydantic.dev/2.11/v/string_type

# new_student = {"age": 20}  # Valid input

# new_student = {"age": "32"} # automatically converts string to int Type coercion

new_student = {"name": "John Doe", "email": "john.doe@example.com", "cgpa": 8.5}

student = Student(**new_student)

print(student)