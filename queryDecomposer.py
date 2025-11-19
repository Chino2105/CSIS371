# Statistical parser
import spacy

# File: queryDecomposer.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/19/2024
# Description: This program decomposes user queries into components
# such as media type, entities, time, and descriptions using spaCy.

# Load the medium English model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spacy English Model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Media Types, Spacy doens't know media types.
media_Types = {
    # Images
    "image", "images",
    "photo", "photos",
    "picture", "pictures",
    "screenshot", "screenshots",
    "graphic", "graphics",
    "drawing", "drawings",
    "illustration", "illustrations",
    "icon", "icons",
    "thumbnail", "thumbnails",

    # Videos
    "video", "videos",
    "movie", "movies",
    "film", "films",
    "clip", "clips",
    "footage",
    "animation", "animations",
    "gif", "gifs",

    # Sound
    "audio",
    "sound", "sounds",
    "music",
    "song", "songs",
    "track", "tracks",
    "podcast", "podcasts",
    "recording", "recordings",
}

def decompose_query(query):

    # group nouns into chunks
    doc = nlp(query)

    media_type = []
    entities = []
    time = []
    descriptions = []

    # Append Time
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            time.append(ent.text)
    # Append Entities
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "WORK_OF_ART"):
            entities.append(ent.text)

    # Append Media Types
    for token in doc:
        if token.text.lower() in media_Types:
            media_type.append(token.text.lower())
    generic_terms = {"something", "someone", "scene", "point", "he", "she", "dude", "people"}

    # Append Descriptions
    for chunk in doc.noun_chunks:
        root = chunk.root.text.lower()
        clean_chunk = chunk.text.lower().replace("a ", "").replace("the ", "")
        if clean_chunk not in time and root not in generic_terms:
            # If not already captured as a named entity, treat as description
            if clean_chunk not in entities:
                descriptions.append(chunk.text)


    descriptions = [d for d in descriptions if d not in entities]
    
    return {
            "media_type": list(set(media_type)),
            "entities": list(set(entities)),
            "time": list(set(time)),
            "descriptions": list(set(descriptions))
        }

# print(decompose_query("That sci-fi movie with the robot kid from the 1990s"))
# print(decompose_query("What's that Stephen King horror movie about a group of kids in a small town? They're all being hunted by a really scary clown that lives in the sewers."))