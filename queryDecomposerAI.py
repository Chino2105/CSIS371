import json
import re

import gpt

# File: queryDecomposerAI.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This program decomposes user queries into structured components
# using a GPT


def call_gpt(query):
    prompt = f"""
    You are an information retrieval assistant. 
    Decompose the following query into a JSON object with these keys:
    - media_type
    - entities
    - attributes
    - time
    - descriptions

    Guidelines:
    - Always return lists for every key (even if empty).
    - Extract only short spans (1–3 words) directly from the query.
    - Do not repeat the full query in any field.
    - Normalize obvious typos (e.g., 'Foriegn Film' → 'foreign film').
    - Prefer distinctive nouns, names, or attributes over generic words.
    - Avoid duplicates across fields.
    - Keep values lowercase unless they are proper names.

    Example:
    Query: "George Clooney movie where a woman witnesses a murder"
    JSON:
    {{
    "media_type": ["movie"],
    "entities": ["George Clooney"],
    "attributes": ["murder"],
    "time": [],
    "descriptions": ["woman witnesses"]
    }}

    Now decompose this query:
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
    # Use this for now so we don't call gpt alot.
    gpt_components = call_gpt(query)
    #print ("GPT Components:", gpt_components)
    return gpt_components

# Example
# query = "That sci-fi movie with the robot kid from the 90s"
# print(decompose_query(query)) - {'media_type': ['movie'], 'entities': ['robot kid'], 'attributes': ['sci-fi'], 'time': ['90s'], 'descriptions': ['sci-fi movie', 'robot kid', '90s']}