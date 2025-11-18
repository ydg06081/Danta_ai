from google import genai
from pydantic import BaseModel
import json
import re

class Function_json(BaseModel):
    language: str
    function_name: str
    function_description: str
    inputs: list[str]
    outputs: list[str]
    function_code: str

def prompt(language, function_name, function_description, inputs, outputs):
    
    return f"""
    Create a function in {language} with the following specifications:
    Function Name: {function_name}
    Description: {function_description}
    Inputs: {inputs}
    Outputs: {outputs}
    make sure to provide only the function code without any additional text.
    don`t use unnecessary logic or libraries. just focus on the function implementation.
    carefully use the line breaks and indentation as per {language} standards. especially for Python, use tab for indentation.
    """

def extract_json_from_response(response_text):
    
    # Find the JSON block using a regular expression
    match = re.search(r'```json(.*?)', response_text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        # If no markdown fences are found, assume the whole text is JSON
        return response_text

def parse_json_output(response_text):
    
    json_text = extract_json_from_response(response_text)
    try:
        function_data = json.loads(json_text)
        # If the response is a list, take the first element
        if isinstance(function_data, list):
            function_data = function_data[0]
        return function_data['function_code']
    except (json.JSONDecodeError, IndexError, KeyError):
        return "Error: Failed to parse the generated code. Please try again."

def generate_function(language, function_name, function_description, inputs, outputs):
    
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt(language, function_name, function_description, inputs, outputs),
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Function_json],
        },
    )

    return parse_json_output(response.text)