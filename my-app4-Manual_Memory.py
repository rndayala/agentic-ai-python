# This program demonstrates how to use the LangChain library to interact with the OpenAI API 
# and obtain responses from the GPT-4o-mini model. It retrieves the OpenAI API key from an environment variable 
# stored in a `.env` file using the `dotenv` library. 
# The program creates an instance of the `ChatOpenAI` class with the specified model 
# and invokes it with multiple messages, including a system message and user messages. 
# Finally, it prints the response received from the model, along with metadata about the model used and token usage.

# invoke the LLM using LangChain with multiple messages,
# providing memory of previous messages in the conversation
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="You are an expert Agentic AI assistant."),
    HumanMessage(content="What is Agentic AI? Please provide a brief explanation.")
]

response = model.invoke(messages)

print("Answer:", response.content)
print("Model used:",  response.response_metadata.get("model_name"))
print("Usage metadata:", response.usage_metadata)


messages.append(response)

messages.append(HumanMessage(content="Now give me first thing I should build?"))

response = model.invoke(messages)

print("Token usage for second response:", response.usage_metadata)