import base64
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# get the image from local path
image_path = Path(__file__).parent / "sample.jpg"

# encode the image to base64? Why?
# Base64 encoding is used to convert the image data into a string format that can be included 
# in the JSON payload for the API request.
encoded = base64.b64encode(
    image_path.read_bytes()
).decode("utf-8")

# create a HumanMessage with the base64-encoded image and a text prompt asking about the animal in the photo.
# base64 encoded data is used to represent binary data (like images) in a text format, 
# which is necessary for sending the image data in a JSON payload to the API.
# Why to specify mime_type? The mime_type is specified to inform the model about the type of image being sent (in this case, a JPEG image).
local_message = HumanMessage(content=[
    {
        "type": "text",
        "text": "What animal is in this photo, and what is it doing?"
    },
    {
        "type": "image",
        "base64": encoded,
        "mime_type": "image/jpeg"
    },
])

print(model.invoke([local_message]).content)