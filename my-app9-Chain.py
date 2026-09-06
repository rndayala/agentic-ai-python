# this program demonstrates how to use LangChain to create a chain that answers a question
# chain - prompt -> model -> output parser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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

# create a chain that first invokes the prompt, then the model, and finally the parser.
chain = prompt | model | parser

# invoke the chain with specific values for the placeholders and print the response content.
print(chain.invoke({
    "limit": 60,
    "question": "What is an API?"
}))

print()

# translate= (ChatPromptTemplate.from_messages([
#         (
#             "human",
#             "Translate this to Hindi:\n\n{text}"
#         )
#     ]) | model | parser )

# translate= (ChatPromptTemplate.from_messages([
#         (
#             "human",
#             "Translate this to Hindi:\n\n{text}"
#         )
#     ]) | model | parser )

# full = chain | translate

# print(
#     full.invoke({
#         "limit": 30,
#         "question": "What is a Java?"
#     })
# )