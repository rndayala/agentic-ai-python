# This Python program is designed to interact with the OpenAI API to obtain a response from the GPT-4o-mini model. 
# It uses the `openai` library to send a request to the API endpoint for chat completions. 
# The program retrieves the OpenAI API key from an environment variable stored in a `.env` file using the `dotenv` library. 
# It sends a prompt asking for a brief explanation of "Agentic AI" and prints the response received from the model.

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="What is Agentic AI? Please provide a brief explanation."
)

print("Answer:", response.output_text)