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

    for chunk in doc.noun_chunks:
        # Append chunck to description
        descriptions.append(chunk.text)

        # Get the main noun and check if it's a media type
        if (chunk.root.text.lower() in media_Types):
            media_type.append(chunk.root.text)
        else:
            # It's a entity
            clean_entity = chunk.text.replace("a ", "").replace("the ", "")

            if clean_entity not in time:
                entities.append(clean_entity)
    
    return {
            "media_type": list(set(media_type)),
            "entities": list(set(entities)),
            "time": list(set(time)),
            "descriptions": list(set(descriptions))
        }

# print(decompose_query("That sci-fi movie with the robot kid from the 1990s"))
# print(decompose_query("What's that Stephen King horror movie about a group of kids in a small town? They're all being hunted by a really scary clown that lives in the sewers."))