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