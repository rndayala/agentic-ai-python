# this program demonstrates how to create an agent that can use tools to answer user questions.
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# create_agent is going to manage the tool-calling loop for us.
from langchain.agents import create_agent
from langchain_core.tools import tool

# load environment variables from a .env file, which may include API keys and other configuration settings.
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

# tool to get the current time in a city. The tool takes a string as input 
# and returns the current time in that city.
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

########################################################
########### AGENT CREATION AND USAGE ###################
########################################################
agent = create_agent(
   model="openai:gpt-4o-mini",  # model in form of provider:model_name
   tools=[
        current_time,
        multiply
   ],
   system_prompt=(
        "You are a helpful assistant. "
        "Use the tools when they fit." 
   ),
)

# asking that agent a question
# message is a list of messages, where each message is a dictionary with a role and content.
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "What time is it in Mumbai, "
                "and what is 98765 times 43210?"
            )
        }
    ]
})

print("Final Answer")
# printing the final answer from the agent's response. 
# The final answer is typically the last message in the list of messages returned by the agent.
print(result["messages"][-1].content)

print()

# printing a summary of what happened during the agent's execution, 
# including the types of messages exchanged and any tools that were called by the agent.
print("What happened along the way:")

for message in result["messages"]:

    # Get the type of message, like HumanMessage, AIMessage, ToolMessage
    #
    kind = type(message).__name__
    if getattr(message, "tool_calls", None):
        # The LLM requested one or more tools.
        # We print the names of those tools.

        print(
            f"  {kind}: asked for "
            f"{[c['name'] for c in message.tool_calls]}"
        )
    else:
        print(
            f"  {kind}: "
            f"{str(message.content)[:150]}"
        )


# OUTPUT:
# Final Answer
# The current time in Mumbai is 08:46 AM on September 7, 2026. The result of 98765 times 43210 is 4,267,635,650.

# What happened along the way:
#   HumanMessage: What time is it in Mumbai, and what is 98765 times 43210?
#   AIMessage: asked for ['current_time', 'multiply']
#   ToolMessage: 07 September 2026, 08:46 AM
#   ToolMessage: 4267635650
#   AIMessage: The current time in Mumbai is 08:46 AM on September 7, 2026. The result of 98765 times 43210 is 4,267,635,650.