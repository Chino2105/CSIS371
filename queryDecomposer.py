import json
import re

import spacy
from spacy.matcher import Matcher

import gpt

# File: queryDecomposer.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This program decomposes user queries into structured components
# using a combination of GPT and spaCy-based rule matching.

# IMPORTANT: run "python -m spacy download en_core_web_sm" to download the spaCy model.

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Rule-based matcher for genres, decades, etc.
matcher = Matcher(nlp.vocab)
matcher.add("GENRE", [[{"LOWER": {"IN": ["sci-fi", "comedy", "drama", "horror"]}}]])
matcher.add("TIME", [[{"LOWER": {"IN": ["90s", "80s", "2000s"]}}]])

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

def normalize_components(components, query):
    """Ensure consistent schema and drop noisy fields."""
    for key in ["media_type", "entities", "attributes", "time", "descriptions"]:
        val = components.get(key, [])
        if isinstance(val, str):
            components[key] = [val]
        elif val is None:
            components[key] = []
    # Drop descriptions that are basically the whole query
    components["descriptions"] = [
        d for d in components["descriptions"]
        if len(d.split()) < len(query.split()) * 0.7
    ]
    return components

def enrich_with_spacy(components, query):
    """Add extra signals from spaCy/rules."""
    doc = nlp(query)
    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end].text
        if label == "GENRE" and span not in components["attributes"]:
            components["attributes"].append(span)
        if label == "TIME" and span not in components["time"]:
            components["time"].append(span)
    # Add adjectives as attributes
    for tok in doc:
        if tok.pos_ == "ADJ" and tok.text not in components["attributes"]:
            components["attributes"].append(tok.text)
    return components

def decompose_query(query):
    gpt_components = call_gpt(query)
    normalized = normalize_components(gpt_components, query)
    enriched = enrich_with_spacy(normalized, query)
    return enriched

# Example
query = "That sci-fi movie with the robot kid from the 90s"
print(decompose_query(query))