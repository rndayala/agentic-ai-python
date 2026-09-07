# this program demonstrates how to use the streaming feature of the agent.
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

# load environment variables from a .env file, which may include API keys and other configuration settings.
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

# =========================================================
# TOOL 1
# =========================================================
# tool to get the flight ticket price in rupees from Mumbai to a city.
@tool
def ticket_price(city: str) -> int:
    """
    Get the flight ticket price in rupees
    from Mumbai to a city.
    """

    prices = {
        "delhi": 4500,
        "bengaluru": 3900,
        "kolkata": 5200
    }

    return prices.get(city.lower(), 6000)


# =========================================================
# TOOL 2
# =========================================================
# tool to get the total hotel cost in rupees for a number of nights in a city.
@tool
def hotel_price(city: str, nights: int) -> int:
    """
    Get the total hotel cost in rupees
    for a number of nights in a city.
    """

    per_night = {
        "delhi": 3000,
        "bengaluru": 3500,
        "kolkata": 2800
    }

    return per_night.get(city.lower(), 3200) * nights

# create an agent that can use the tools to answer questions about travel costs.
agent = create_agent(
   model="openai:gpt-4o-mini", 
   tools=[
        ticket_price,
        hotel_price
   ],
   system_prompt=(
        "You plan small trips and always look up "
        "real numbers with the tools."
   ),
)

# we are going to ask the agent a question about the total cost of a trip 
# from Mumbai to Bengaluru for 3 nights.
question = (
    "I want to go to Bengaluru for 3 nights "
    "from Mumbai. What is my total cost?"
)

# we are going to stream the agent's response to the question,
# which means we will receive updates as the agent processes the question and uses the tools.

# stream takes input in the form of a dictionary with a "messages" key, which is a list of messages.
# stream_mode="updates" tells the agent to give us updates as each step happens, 
# rather than waiting until the end to give us a final answer.

# stream returns an iterator that yields chunks of updates from the agent. 
# Each chunk is a dictionary with nodes and updates.
for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    },
    stream_mode="updates"   # this is to tell our agent give me update as each step happens
):
    for node, update in chunk.items():
        for message in update["messages"]:
            # if the message contains tool calls, we print the names of those tools.
            if getattr(message, "tool_calls", None):
                print(
                    f"[{node}] wants: "
                    f"{[c['name'] for c in message.tool_calls]}"
                )

            # if the message is of type ToolMessage, we print the name of the tool and its output.
            elif type(message).__name__ == "ToolMessage":
                print(
                    f"[{node}] "
                    f"{message.name} returned: "
                    f"{message.content}"
                )
            
            # if the message doesn't contain tool_calls and it's not ToolMessage,
            # that indicates that the message is a final answer from the agent, so we print it. 
            elif message.content:
                print(
                    f"[{node}] says: "
                    f"{message.content}"
                )

print("-" * 60)