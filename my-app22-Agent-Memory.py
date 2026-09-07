# this program demonstrates how to create an agent with memory using LangChain.
# InMemorySaver is used to store the conversation after each step, 
# allowing the agent to remember previous interactions.
# However this is not a persistent memory, so if you restart the program, the memory will be lost.
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# load environment variables from a .env file, which may include API keys and other configuration settings.
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

# tool to book a seat in the class for a person. 
# The tool takes a name and a seat number as input and returns a confirmation message.
@tool
def book_seat(name: str, seat: str) -> str:
    """Book a seat in the class for a person."""
    return f"Seat {seat} booked for {name}."

# create an agent that can use the book_seat tool to answer questions about booking seats.
# We are creating agent with memory using InMemorySaver, 
# which allows the agent to remember previous interactions.
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[book_seat],
    system_prompt="You help learners book a seat in the LangChain class.",
    checkpointer=InMemorySaver(),  # this tell agent to store conversation after each step
)

# function to ask a question to the agent and print the response.
# while invoking the agent, we are passing a thread_id to the config,
# which allows the agent to maintain separate conversations for different threads.
def ask(question, thread_id):
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config=
        {
            "configurable": {
                "thread_id": thread_id
            }
        },
    )
    print(f"[{thread_id}] Q: {question}")
    print(f"[{thread_id}] A: {result['messages'][-1].content}")
    print()

# asking the agent a series of questions in two different threads, "vinay" and "nimish"
# asking the agent in thread "vinay" to remember the name and seat booked
ask("My name is vinay kumar gurram", thread_id="vinay")
ask("Book me a seat A12.", thread_id="vinay")
ask("What is my name and what seat did i book", thread_id="vinay")

# asking the agent in thread "nimish" to remember the name and seat booked
ask("What is my name and what seat did i book", thread_id="nimish")

