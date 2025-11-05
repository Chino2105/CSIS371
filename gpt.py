import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() # Load Envirmonemtal Vars

client = OpenAI(api_key=os.getenv("GPT_KEY")) # Load OpenAI

def sendGPT(text):
    response = client.responses.create(
        model="gpt-5-nano",
        input=text
    )
    
    return response.output_text
