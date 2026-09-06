# this program demonstrates how to use LangChain to stream responses from a model and a chain of prompts.
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# create a model instance with the specified model and temperature.
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# streaming diretly from model
for chunk in model.stream(
    "write a four line poem about debugging at mid night"):
    print(chunk.content, end="", flush=True)

print("\n" + "-" * 50)

# create chain from Prompt, Model and Output Parser
chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Explain {topic} in about 100 words."
            )
        ]
    )
    | model
    | StrOutputParser()
)

# streaming from chain
# provide the topic to the chain and stream the output in chunks, printing each chunk as it is received.
for piece in chain.stream(
    {
        "topic": "how HTTP works"
    }
    ):
    print(piece, end="", flush=True)

