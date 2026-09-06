# this program demonstrates how to use LangChain to create a chain that answers a question
# and translates the answer into Hindi.
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# reconfigure the standard output to use UTF-8 encoding to ensure that 
# any non-ASCII characters (like those in Hindi) are displayed correctly in the console.
sys.stdout.reconfigure(encoding="utf-8")

# create a chat prompt template with system and human messages, using placeholders for dynamic content.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful tutor. Answer in {limit} words or less."
    ),
    (
        "human",
        "{question}"
    ),
])

# create a model instance with the specified model and temperature.
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# create an instance of the StrOutputParser class to parse the model's output into a string format.
parser = StrOutputParser()

# First chain : create a chain that first invokes the prompt, then the model, and finally the parser.
chain = prompt | model | parser

# Second chain : create a chain for translation.
translate= (ChatPromptTemplate.from_messages([
        (
            "human",
            "Translate this to Hindi:\n\n{text}"
        )
    ]) | model | parser )


# chain of chains
# Full chain : create a full chain that first answers the question and then translates the answer into Hindi.
full = chain | translate

print(
    full.invoke({
        "limit": 30,
        "question": "What is a Java?"
    })
)