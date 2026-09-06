import base64
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# get the image from image url
image_url= "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg" 

# create a HumanMessage with the image URL and a text prompt asking about the picture.
# image URL is used to provide the model with a reference to the image, 
# allowing it to analyze and describe the content of the image.
url_message = HumanMessage(content=[
    {
        "type": "text",
        "text": "Describe this picture in two sentences. Also talk about breed of this "
    },
    {
        "type": "image",
        "url": image_url
    },
])

# invoke the model with the HumanMessage containing the image URL and print the response content.
print(model.invoke([url_message]).content)
