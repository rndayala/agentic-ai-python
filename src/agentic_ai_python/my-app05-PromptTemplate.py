# This program demonstrates how to use the LangChain library to create a chat prompt template and invoke the model.
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Create an instance of the ChatOpenAI class with the specified model and temperature.
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
# Create a chat prompt template with system and human messages, using placeholders for dynamic content.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a {language} trainer. Keep answers under {limit} words"
        ),
        (
            "human",
            "Explain {topic} to a beginner."
        )
    ]
)
# filling the prompt template with specific values for the placeholders
filled = prompt.invoke({
    "language": "Java",
    "limit": 60,
    "topic": "JDBC"
})
print(filled.messages)
print()

# Invoke the model with the filled prompt and print the response content.
response = model.invoke(filled)
print(response.content)