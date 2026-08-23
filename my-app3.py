# This Python program is designed to interact with the OpenAI API using the LangChain library 
# to obtain a response from the GPT-4o-mini model. It retrieves the OpenAI API key from an environment variable 
# stored in a `.env` file using the `dotenv` library. The program creates an instance of the `ChatOpenAI` class 
# with the specified model and invokes it with a prompt asking for a brief explanation of "Agentic AI." 
# Finally, it prints the response received from the model.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

# invoke the llm
response = model.invoke("What is Agentic AI in short ?")

print("Answer:", response.content)