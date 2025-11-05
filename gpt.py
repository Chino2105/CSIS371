import os

from dotenv import load_dotenv
from openai import OpenAI

# File: gpt.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This program serves as a wrapper for OpenAI's GPT API to facilitate

load_dotenv() # Load Envirmonemtal Vars

client = OpenAI(api_key=os.getenv("GPT_KEY")) # Load OpenAI

def sendGPT(text):
    response = client.responses.create(
        model="gpt-5-nano",
        input=text
    )
    
    return response.output_text
