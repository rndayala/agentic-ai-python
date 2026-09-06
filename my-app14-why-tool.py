# this program demonstrates the problems with model and why tool calling is required
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# create a model instance with the specified model.
model = ChatOpenAI(model="gpt-4o-mini")

print("Q1: What is the time right now in Mumbai")

# each model has a training cut-off date, and the model may not have access to real-time information.
# invoke the model with a prompt and print the response content.
print(model.invoke("What is the time right now in Mumbai").content)
print("-" * 60)

# some times, models may not be able to perform simple calculations accurately, as they are not designed to be calculators.
# the result from model and python calculation may differ, 
# so it is important to use external tools for accurate calculations.
print("Q2: What is 98765 multiplied by 43210?")
print(model.invoke("What is 98765 multiplied by 43210?").content)

print("The python calc : ",98765 * 43210 )