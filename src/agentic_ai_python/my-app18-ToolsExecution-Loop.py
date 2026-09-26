# this program demonstrates the execution of tools by a language model in a conversational context using LangChain.
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Just makes the console support UTF-8 characters.
sys.stdout.reconfigure(encoding="utf-8")

#Create our tools

# tool to get the current time in a city. 
# The tool takes a string as input and returns the current time in that city.
@tool
def current_time(city: str) -> str:
    """Get the current time in a city."""

    zones = {
        "mumbai": "Asia/Kolkata",
        "london": "Europe/London",
        "new york": "America/New_York"
    }

    zone = zones.get(city.lower())

    if zone is None:
        return f"I do not know the timezone for {city}."

    return datetime.now(
        ZoneInfo(zone)
    ).strftime("%d %B %Y, %I:%M %p")

# tool to multiply two numbers. The tool takes two integers as input and returns their product.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the exact result."""

    return a * b

# put our tools into a list
tools = [
    current_time,
    multiply
]

# we need to use name of tool by llm to find actual python tool 
tools_by_name = {
    t.name: t for t in tools
}

# create model and bind the tools to it. The model will be able to invoke the tools based on user input.
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools(tools)

# create the conversation
messages = [
    # Tell the LLM what kind of assistant it is.
    SystemMessage(
        content="You are a helpful assistant. Use the tools when they fit."
    ),

    # This is the user's actual question.
    HumanMessage(
        content="What time is it in Mumbai, and what is "
                "98765 times 43210?"
    ),
]

#start tool calling loop 

step = 1

while True:
    response = model.invoke(messages)
    # print(f"\nStep {step}: model response : {response}")
    messages.append(response)

    if not response.tool_calls:
        print("\nFinal answer : ")
        print(response.content)
        break

    print(
        f"Step {step}: the model asked for "
        f"{len(response.tool_calls)} tool call(s)" 
    )

    for call in response.tool_calls:

        # call contains: tool name , arguments, tool call id

        tool_to_run = tools_by_name[call["name"]]

        tool_message = tool_to_run.invoke(call)
        
        print(
            f"   {call['name']}({call['args']}) "
            f"-> {tool_message.content}"
        )

        messages.append(tool_message)

step +=1

# print the messages in the conversation, showing the types of messages exchanged between the user, model, and tools.
print(
    "\nMessages in the conversation:",
    [type(m).__name__ for m in messages]
)

