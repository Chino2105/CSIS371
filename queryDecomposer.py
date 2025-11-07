import json
import re

import gpt

# File: queryDecomposer.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This program decomposes user queries into structured components
# using a GPT


def call_gpt(query):
    """Call your GPT wrapper with a structured prompt."""
    prompt = f"""
    Decompose the query into JSON with keys: media_type, entities, attributes, time, descriptions.
    - Always return lists (even if empty).
    - Only extract short spans directly from the query (1–3 words).
    - Do not repeat the full query in any field.
    Query: "{query}"
    JSON:
    """
    response = gpt.sendGPT(prompt)  # your GPT call
    # Strip code fences if present
    response = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"media_type": [], "entities": [], "attributes": [], "time": [], "descriptions": []}

def decompose_query(query):
    gpt_components = call_gpt(query)
    return gpt_components

# Example
query = "That sci-fi movie with the robot kid from the 90s"
print(decompose_query(query))