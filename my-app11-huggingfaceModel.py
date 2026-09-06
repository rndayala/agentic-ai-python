# this program demonstrates how to use HuggingFace endpoint to invoke a model and get a response.
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# create a HuggingFace endpoint instance with the specified model and parameters.
endpoint= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text generation",
    max_new_tokens=200,
    temperature=0.3,
)

# create a model instance using the HuggingFace endpoint.
model = ChatHuggingFace(llm=endpoint)

# invoke the model with a prompt and print the response content.
print(model.invoke("Explain what an open weights model is, in two lines").content)