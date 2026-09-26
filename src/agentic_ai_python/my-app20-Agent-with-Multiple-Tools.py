# this program demonstrates how to create an agent with multiple tools, 
# and how to use it to answer questions about orders, stock, pricing, and delivery.
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

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
        "amount": 1499
    },

    "ORD-1002": {
        "item": "Mechanical keyboard",
        "status": "packed",
        "amount": 4999
    },
}

# STOCK
# this dictionary represents the stock of items in a warehouse,
# where the keys are item names and the values are the number of units available.
STOCK = {
    "wireless mouse": 12,
    "mechanical keyboard": 0,
    "usb hub": 34
}

# our tools for this app

# =========================================================
# TOOL 1
# =========================================================

# tool to get the status and amount of an order using its id.
@tool
def order_status(order_id: str) -> str:
    """
    Get the status and amount of an order using its id,
    for example ORD-1001.
    """

    order = ORDERS.get(order_id.upper())

    if order is None:
        return f"No order found with id {order_id}."

    return (
        f"{order['item']}, "
        f"status {order['status']}, "
        f"amount {order['amount']} rupees"
    )


# =========================================================
# TOOL 2
# =========================================================
# tool to check how many units of an item are left in the warehouse.
@tool
def check_stock(item: str) -> str:
    """
    Check how many units of an item are left in the warehouse.
    """

    count = STOCK.get(item.lower())

    if count is None:
        return f"{item} is not in the catalogue."

    return f"{count} units of {item} in stock"


# =========================================================
# TOOL 3
# =========================================================
# tool to apply a discount percentage to an amount and return the new amount.
@tool
def apply_discount(amount: float, percent: float) -> float:
    """
    Apply a discount percentage to an amount
    and return the new amount.
    """

    return round(
        amount - (amount * percent / 100),
        2
    )


# =========================================================
# TOOL 4
# =========================================================
# tool to estimate delivery time for an Indian pin code.
@tool
def delivery_days(pin_code: str) -> str:
    """
    Estimate delivery time for an Indian pin code.
    """

    metro = {
        "400001",
        "110001",
        "560001"
    }

    return "2 days" if pin_code in metro else "5 days"

# CREATE AN AGENT
# The agent is created using the create_agent function, 
# which takes the model name, a list of tools, and a system prompt as arguments. 
# The system prompt provides context to the agent about its role and how it should use the tools.
agent = create_agent(
   model="openai:gpt-4o-mini", 
   tools=[
        order_status,
        check_stock,
        apply_discount,
        delivery_days
   ],
   system_prompt=(
        "You are a support assistant for an online store. "
        "Use the tools for anything about orders, stock, "
        "pricing or delivery"
        "Never guess a number, look it up."
   ),
)

# Helper function
# The ask function is a helper function that takes a question as input,
# invokes the agent with that question, and prints the agent's response.
def ask(question):
    # Print the user's question.
    print("Q:", question)

    # Invoke the agent with the user's question
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    # Print the agent's final answer, which is typically the last message in the list of messages 
    # returned by the agent.
    print(
        "A:",
        result["messages"][-1].content
    )


    print("-" * 60)

# List of questions to ask the agent.
# Each question is designed to test the agent's ability to use the tools effectively.
ask("What is the status of order ORD-1001")
ask("Do you have a mechanical keyboard in stock")
ask("when the mechanical keyboard will be available again ?")
ask(
    "For order ORD-1002, what would the price be "
    "after a 10 percent discount?"
)

