# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__),'.env')
load_dotenv(dotenv_path=env_path)

def generate(input_text: str | list[str] = "안녕하세요"):
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    if isinstance(input_text, str):
        input_texts = [input_text]
    else:
        input_texts = input_text

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"{input_text}"),
            ],
        ),
    ]
    

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
