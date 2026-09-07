# this program demonstrates how to create an agent that can 
# handle customer support queries for an online store.

# In this program, we generate structured output from the agent, which includes 
# information about the order, the customer's intent, and what action was taken by the agent.
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

# pydantic model for tool input to make sure we are not blindly trust argument
class RefundInput(BaseModel):

    # The order ID is required.
    order_id: str = Field(
        description="Order id, for example ORD-1001"
    )

    # The reason is required and must contain at least 5 characters.
    reason: str = Field(
        description="Why the customer wants a refund",
        min_length=5
    )


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

# ============================================================
# TOOL 2 - DELIVERY ESTIMATE
# ============================================================
# This tool takes an order ID as input and returns an estimated delivery time for the order.
@tool
def delivery_estimate(order_id: str) -> str:
    """Estimate when an order will be delivered."""

    order = ORDERS.get(order_id.upper())

    if order is None:
        return f"No order found with id {order_id}."

    # Our demo logic:
    # PIN starting with 4 -> 2 days
    # Otherwise           -> 5 days
    #
    # In a real application this could call a delivery API.

    return "2 days" if order["pin"].startswith("4") else "5 days"

# ============================================================
# TOOL 3 - START REFUND
# ============================================================
# This tool takes an order ID and a reason as input and starts a refund process for the order.
@tool("start_refund", args_schema=RefundInput)
def start_refund(order_id: str, reason: str) -> str:
    """
    Start a refund for an order.

    Only use it after the customer clearly asks for one.
    """

    # Check whether the order exists.
    if order_id.upper() not in ORDERS:
        return f"Cannot refund, no order with id {order_id}."

    # If the order exists, start the refund.
    return (
        f"Refund started for {order_id.upper()}, "
        f"reason recorded as: {reason}"
    )


# ============================================================
# TOOL 4 - ESCALATE TO HUMAN
# ============================================================
# This tool sends a case to a human agent when the AI cannot solve it.
@tool
def escalate(order_id: str, note: str) -> str:
    """Send the case to a human agent when you cannot solve it."""

    return (
        f"Case for {order_id} sent to a human "
        f"with note: {note}"
    )

# ============================================================
# STRUCTURED OUTPUT MODEL
# ============================================================
#
# For example, instead of only:
#
#     "Your order is packed."
#
# we can get:
#
#     order_id     -> ORD-1002
#     intent       -> status
#     message      -> Your order is packed.
#     action_taken -> Checked order status
#     needs_human  -> False

# class for structured output from the agent, which includes 
# information about the order, the customer's intent, and what action was taken by the agent.
class Reply(BaseModel): 
    """Structured result returned by the support agent.""" 

    # Which order was discussed?
    order_id: str = Field( description="Order discussed, or NA" )

    intent: Literal[ 
        "status",
        "delivery", 
        "refund", 
        "other" ] = Field( description="What the customer wanted" )
    message: str = Field( description="The reply to show the customer" )

    action_taken: str = Field( description="What the agent actually did" )
    needs_human: bool = Field( description="True if a person must follow up" )


# Create the agent with the tools and structured output model.
# response_format=Reply : tells the agent to return structured output in the form of the Reply model defined above.
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        order_status,
        delivery_estimate,
        start_refund,
        escalate
    ],
    system_prompt=(
        "You are the support agent for an online store. "
        "Always look up an order before answering about it, never guess numbers or dates. "
        "Start a refund only when the customer asks for one. "
        "Escalate anything you cannot handle with the tools you have. "
        "In the final response, clearly state the order id and what action was taken."
    ),
    response_format=Reply,
    checkpointer=InMemorySaver(),
)

# Define a conversation with the customer
conversation = [
    "Hi, where is my order ORD-1002?",
    "When will it reach me?",
    "That is too late, I want a refund because I need it this week.",
]

# config for the agent to maintain a separate conversation thread for this customer.
config = {
    "configurable": {
        "thread_id": "telusko-new"
    }
}

# why now agent generates structured output instead of just text, 
# which allows us to extract specific information from the agent's response?
# While creating the agent, we specified response_format=Reply, 
# which tells the agent to return structured output in the form of the Reply model defined above.
for question in conversation:
    result =agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config=config
    )
    reply =result["structured_response"]
    
    print("Customer: ", question)
    print("Agent: ", reply.message)
    print()
    print("Application data:")
    print(f" order_id = {reply.order_id}")
    print(f" intent = {reply.intent}")
    print(f" action_taken = {reply.action_taken}")
    print(f" needs_human = {reply.needs_human}")
    print("-" * 70)

