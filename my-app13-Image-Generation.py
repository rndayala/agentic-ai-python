import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from pathlib import Path

# load environment variables from a .env file, which may include API keys and other configuration settings.
load_dotenv()

# client open ai 
client = OpenAI()

# path to the current directory, used for saving generated images.
HERE = Path(__file__).parent

# save function to decode the base64-encoded image data and save it to a file.
def save(result, filename):

    path = HERE / filename

    path.write_bytes(
        base64.b64decode(result.data[0].b64_json)
    )

    print("Saved:", path.name)

# create writer chain that generates a short, visual image prompt based on a given topic.
writer = (
    ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write short, visual image prompts. One sentence, no preamble."
        ),
        (
            "human",
            "An illustration for a blog post about {topic}"
        )
    ])
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# generate an image prompt for the topic "learning LangChain" using the writer chain.
image_prompt =writer.invoke({
    "topic": "learning LangChain"
})

# print the generated image prompt and 
# then use the OpenAI client to generate an image based on that prompt, saving the result to a file named "telusko.png".
print("Our Chain Write this : ", image_prompt)

# generate an image using the OpenAI client with the specified model, prompt, size, and quality settings.
result= client.images.generate(
    model="gpt-image-1-mini",
    prompt=image_prompt,
    size="1024x1024",
    quality="low",
)

# call the save function to decode the base64-encoded image data and save it to a file named "telusko.png".
save(result, "telusko.png")