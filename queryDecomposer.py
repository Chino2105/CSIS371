# Statistical parser
import spacy

# File: queryDecomposer.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/20/2025
# Description: This program decomposes user queries into components
# such as media type, entities, time, and descriptions using spaCy.

# Load the medium English model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy English Model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

media_Types = {
    "image","images","photo","photos","picture","pictures","screenshot","screenshots",
    "graphic","graphics","drawing","drawings","illustration","illustrations","icon","icons",
    "thumbnail","thumbnails","video","videos","movie","movies","film","films","clip","clips",
    "footage","animation","animations","gif","gifs","audio","sound","sounds","music","song",
    "songs","track","tracks","podcast","podcasts","recording","recordings"
}

# Clean and deduplicate entities
def clean_entities(entities):
    entities = [e.strip() for e in entities]
    final = []
    for e in entities:
        # Skip single names unless no longer form exists
        if len(e.split()) == 1:
            continue
        # Normalize suffixes like "Movies" or "Films"
        if e.lower().endswith((" movie"," movies"," film"," films")):
            e = " ".join(e.split()[:-1])
        # Keep only if not a substring of another entity
        if not any(e.lower() in other.lower() and e.lower() != other.lower() for other in entities):
            final.append(e.title())
    return list(set(final))

# Decompose the query into components
def decompose_query(query):
    doc = nlp(query)

    media_type, entities, time, descriptions = [], [], [], []

    # Time
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            time.append(ent.text)
        if ent.label_ in ("PERSON", "ORG", "GPE", "WORK_OF_ART"):
            entities.append(ent.text)

    # Media types
    for token in doc:
        if token.text.lower() in media_Types:
            media_type.append(token.text.lower())

    # Short noun phrases
    for chunk in doc.noun_chunks:
        words = chunk.text.split()
        if 2 <= len(words) <= 5 and chunk.root.text.lower() not in {"something","someone","scene","point","he","she","dude","people"}:
            descriptions.append(chunk.text)

    # Verb–object phrases
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            obj = [child.text for child in token.head.children if child.dep_ in ("dobj","attr","oprd")]
            if obj:
                phrase = f"{token.text} {token.head.text} {obj[0]}"
                descriptions.append(phrase)

    # Clean entities and deduplicate descriptions
    entities = clean_entities(entities)
    descriptions = [d for d in set(descriptions) if 2 <= len(d.split()) <= 5]

    return {
        "media_type": sorted(set(media_type)),
        "entities": sorted(set(entities)),
        "time": sorted(set(time)),
        "descriptions": sorted(set(descriptions))
    }

# Example test
# if __name__ == "__main__":
#     q = "George Clooney movie where a woman witnesses a murder through the window and he goes to trial"
#     print(decompose_query(q))