# This program demonstrates how to create and use tools in LangChain. 
# Tools are functions that can be invoked by the model to perform specific tasks. 
# In this example, we define two tools: one for getting the current time in a city and 
# another for multiplying two numbers.

from langchain_core.tools import tool
from datetime import datetime
from zoneinfo import ZoneInfo

# creating a tool to get the current time in a city. The tool takes a city name as input 
# and returns the current time in that city.
@tool
def current_time(city: str) -> str:
    """Get the current time in a city. Use it whenever the user asks about time."""
    zones = {
        "mumbai": "Asia/Kolkata",
        "delhi": "Asia/Kolkata",
        "london": "Europe/London",
        "new york": "America/New_York",
    }
    zone = zones.get(city.lower())
    if zone is None:
        return f"I do not know the timezone for {city}."
    return datetime.now(ZoneInfo(zone)).strftime("%d %B %Y, %I:%M %p")

# creating a tool to multiply two numbers. The tool takes two integers as input and returns their product.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the exact result."""
    return a * b


# Displaying tools information
# Each tool will have name, args, description and docstring. 
# The model can use this information to understand what the tool does and how to invoke it.
print("name : ", current_time.name)
print("description : ", current_time.description)
print("args :", current_time.args)
print()
print("name : ", multiply.name)
print("description : ", multiply.description)
print("args :", multiply.args)
print()

# invoking the current_time tool with a city name and printing the result.
# Note: here we are not invoking from the model, but directly calling the tool function.
print("Current time in Mumbai:", current_time.invoke({"city": "Mumbai"}))

# invoking the multiply tool with two numbers and printing the result.
# Note: here we are not invoking from the model, but directly calling the tool function. 
# In a real application, the model would invoke the tool based on user input.
response = multiply.invoke({"a": 98765, "b": 43210})
print("Product of 98765 and 43210:", response)




