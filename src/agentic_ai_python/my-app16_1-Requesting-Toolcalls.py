# this program demonstrates how to create and use tools in LangChain.
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# load environment variables from a .env file, which may include API keys and other configuration settings.
load_dotenv()

# reconfigure stdout to use UTF-8 encoding to ensure that any output from 
# the program is correctly encoded and displayed, especially when dealing with non-ASCII characters.
sys.stdout.reconfigure(encoding="utf-8")

# create a model instance using langchain_openai's ChatOpenAI class, 
# specifying the model to be used for generating responses.
model=ChatOpenAI(model="gpt-4o-mini")

# create tool for multiplying two numbers.
# The tool takes two integers as input and returns their product.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the exact result."""
    return a * b

# create tool for counting the number of words in a given text.
# The tool takes a string as input and returns the word count.
@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())

# bind the tools to the model using the bind_tools method.
model_with_tools=model.bind_tools([
    multiply,
    word_count
])

# for this request, the model will decide to use the multiply tool
response =model_with_tools.invoke("What is 98765 mulitplied by 43210?")
print("content : ", repr(response.content))
print("tool calls: ", response.tool_calls)

# for this request, the model will not use any tools, as it can answer the question directly 
# without needing to invoke a tool.
chat = model_with_tools.invoke("give me one line on why Python is famous for AI")
print("content : ", chat.content)
print("tool calls: ", chat.tool_calls)


# for this request, the model will decide to use the word_count tool 
# to count the number of words in the given sentence.
reply = model_with_tools.invoke(
    "How many words are in the sentence : "
    "LangChain makes agents simple"
)
print("content : ", reply.content)
print("tool calls: ", reply.tool_calls)