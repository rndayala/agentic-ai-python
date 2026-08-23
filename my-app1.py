# This Python program is designed to interact with the OpenAI API to obtain a response from the GPT-4o-mini model.
# It uses the `requests` library to send a POST request to the API endpoint for chat completions. 
# The program retrieves the OpenAI API key from an environment variable stored in a `.env` file using the `dotenv` library.

import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "What is Agentic AI? Please provide a brief explanation."}
    ]
}

# invoke the llm
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    answer = data['choices'][0]['message']['content']
    print("Answer:", answer)
else:
    print("Error:", response.status_code, response.text)

