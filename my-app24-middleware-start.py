# this program demonstrates how to create an agent that can 
# handle customer support queries for an online store.

# In this program, we create an agent and ask it a question about an order. 
# The agent uses a tool to look up the order status.
import sys

from typing import Literal
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# pydantic is used to defined structured output models and validate tool inputs.
from pydantic import BaseModel, Field 

# load environment variables from a .env file, which may include API keys and other configuration settings.
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

# DATA
# typically this data would be stored in a database, but for the sake of this example,
# we are using a simple dictionary to represent it.
ORDERS = {
    "ORD-1001": {
        "item": "Wireless mouse",
        "status": "shipped",
        "amount": 1499,
        "pin": "400001"
    },
    "ORD-1002": {
        "item": "Mechanical keyboard",
        "status": "packed",
        "amount": 4999,
        "pin": "560034"
    },
}


# tools

# ============================================================
# TOOL 1 - CHECK ORDER STATUS
# ============================================================
# This tool takes an order ID as input and returns the item, status, and amount of the order.
@tool
def order_status(order_id: str) -> str:
    """Get the item, status and amount of an order using its id."""
    order = ORDERS.get(order_id.upper())

    # If the order does not exist, return a useful message.
    if order is None:
        return f"No order found with id {order_id}."

    # Return the information that the agent needs.
    return (
        f"{order['item']}, "
        f"status {order['status']}, "
        f"amount {order['amount']} rupees"
    )

# Create the agent with the tools
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        order_status
    ],
    system_prompt=(
        "You are the support agent for an online store. "
        "Always look up an order before answering about it, never guess numbers or dates. "
    ),
)

# Define a conversation with the customer
conversation = [
    "Hi, where is my order ORD-1002?"
]

# invoke the agent with the conversation and print the response
result = agent.invoke({ "messages": conversation })

# print the response from the agent, which is the last message in the conversation.
print("Agent response:", result["messages"][-1].content)