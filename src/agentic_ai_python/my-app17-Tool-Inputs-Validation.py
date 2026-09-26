# this program demonstrates validating tool inputs using Pydantic models and the LangChain library.
# Validate the arguments passed to a tool using Pydantic models. The tool will only accept valid inputs, 
# and will reject invalid inputs with an error message.

from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field # for validating tool inputs
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Class to validate the input arguments for the currency conversion tool. 
# It uses Pydantic's BaseModel to define the expected structure and constraints of the input data.
class ConvertInput(BaseModel):
    """Input required by the currency conversion tool."""

    # amount is a field that must be a float and greater than 0. 
    # The description provides context for the user.
    # If the negative value is passed, it will be rejected with an error message.
    amount: float = Field(
        description="How many rupees to convert",
        gt=0
    )

    # Only these three values are allowed.
    # JPY, INR, etc. will be rejected.
    currency: Literal["USD", "EUR", "GBP"] = Field(
        description="Currency to convert into"
    )



# create tools
# creating tool to convert an amount in Indian rupees into another currency. The tool takes two arguments:
# amount (float) and currency (Literal["USD", "EUR", "GBP"]). 
# The tool will only accept valid inputs, and will reject invalid inputs with an error message.

# args_schema=ConvertInput is used to validate the input arguments using the Pydantic model defined above.
@tool(
    "convert_from_rupees", # name of the tool
    args_schema=ConvertInput # schema for validating the input arguments using Pydantic model
)
def convert_from_rupees(amount: float, currency: str) -> str:
    """Convert an amount in Indian rupees into another currency."""

    rates = {
        "USD": 0.012,
        "EUR": 0.011,
        "GBP": 0.0094
    }

    return f"{amount} rupees is about {amount * rates[currency]:.2f} {currency}"

# Details about the tool
# each tool will have name, args, description.
print("name", convert_from_rupees.name)
print("description", convert_from_rupees.description)
print("args", convert_from_rupees.args)

print()

# invoking the convert_from_rupees tool with valid input arguments and printing the result.
print(
    convert_from_rupees.invoke(
        {
            "amount": 5000,
            "currency": "USD"
        }
    )
)

print()

# invoking the convert_from_rupees tool with invalid input arguments and printing the error message.
for bad_input in [
    {"amount": -100, "currency": "USD"},
    {"amount": 500, "currency": "JPY"}
]:
    try:
        convert_from_rupees.invoke(bad_input)

    except Exception as error:
        print(
            "Rejected",
            bad_input,
            "because of",
            type(error).__name__
        )

print()

# create a model instance with the specified model.
model=ChatOpenAI(model="gpt-4o-mini")

# bind the tool to the model, so that the model can invoke the tool when needed.
# binding does not mean that the model will always use the tool, but it can use the tool 
# when it needs to perform a task that requires the tool's functionality.
model_with_tools=model.bind_tools(
    [
        convert_from_rupees
    ]
)

# invoking the model with a prompt that requires the use of the convert_from_rupees tool.
response = model_with_tools.invoke("Convert 1000 rupees into EUR.")
print("For request: Convert 1000 rupees into EUR.")
print("tool calls:", response.tool_calls)
# User --> LLM --> LLM decides: "I need convert_from_rupees" --> Tool call generated 
# --> AI application executes convert_from_rupees() --> Tool result --> LLM receives tool result --> LLM generates final response

print()

# invoking the model with a prompt that doesn't require the use of the convert_from_rupees tool.
response = model_with_tools.invoke("What is the capital of France?")
print("For request: What is the capital of France?")
print("tool calls:", response.tool_calls)
print(response.content)

