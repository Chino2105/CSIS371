# Improved Statistical parser with refinements
import re

import spacy
from spacy.matcher import Matcher

# File: queryDecomposerImproved.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 12/02/2025
# Description: Decomposes user queries into components
# such as media type, entities, time, descriptions, and events using spaCy.
# Made for improved accuracy from previous version.

# Load the medium English model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy English Model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Add a sentencizer if not present
if 'sentencizer' not in nlp.pipe_names:
    try:
        nlp.add_pipe('sentencizer')
    except Exception:
        pass

media_Types = {
    "image","images","photo","photos","picture","pictures","screenshot","screenshots",
    "graphic","graphics","drawing","drawings","illustration","illustrations","icon","icons",
    "thumbnail","thumbnails","video","videos","movie","movies","film","films","clip","clips",
    "footage","animation","animations","gif","gifs","audio","sound","sounds","music","song",
    "songs","track","tracks","podcast","podcasts","recording","recordings"
}

# Clean and deduplicate entities
def clean_entities(entities, doc):
    entities = [e.strip() for e in entities if e.strip()]
    final = []
    for e in entities:
        if not e:
            continue
        # Trim suffixes like "Movies" or "Films"
        if e.lower().endswith((" movie"," movies"," film"," films")):
            e = " ".join(e.split()[:-1])
        # Accept single-token entities only if proper noun
        if len(e.split()) == 1:
            for token in doc:
                if token.text == e and token.pos_ == "PROPN":
                    final.append(e.title())
                    break
            continue
        # Keep if not a substring of another
        if not any(e.lower() in other.lower() and e.lower() != other.lower() for other in entities):
            final.append(e.title())
    return list(dict.fromkeys(final))

# Regex fallback for capitalized names if spaCy misses PERSON
def regex_entities(query):
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query)

# Decompose the query into components
def decompose_query(query):
    doc = nlp(query)

    media_type, entities, time, descriptions, events = [], [], [], [], []

    # Named entities
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            time.append(ent.text)
        if ent.label_ in ("PERSON", "ORG", "GPE", "WORK_OF_ART"):
            entities.append(ent.text)

    # Fallback if no entities found
    if not entities:
        entities.extend(regex_entities(query))

    # Media types
    for token in doc:
        if token.text.lower() in media_Types:
            media_type.append(token.text.lower())

    # Short noun phrases
    stop_chunks = {"something","someone","scene","point","dude","people"}
    for chunk in doc.noun_chunks:
        words = chunk.text.split()
        if 2 <= len(words) <= 7 and chunk.root.text.lower() not in stop_chunks:
            clean = re.sub(r'^(a|the|this|that)\s+', '', chunk.text.lower())
            descriptions.append(clean)

    # Verb-centered event extraction with pronoun resolution
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == 'VERB':
                subj, dobj, prep_phrases = None, None, []
                negated = any(ch.dep_ == 'neg' for ch in token.children)
                for ch in token.children:
                    if ch.dep_ in ('nsubj', 'nsubjpass'):
                        subj = ch.text
                    if ch.dep_ in ('dobj', 'attr', 'oprd'):
                        dobj = ch.text
                    if ch.dep_ == 'prep':
                        pobj = ' '.join([t.text for t in ch.subtree])
                        prep_phrases.append(pobj)

                # Pronoun resolution across sentence and previous sentences
                if subj and subj.lower() in {'he','she','they','her','him'}:
                    resolved = None
                    for prev in reversed(list(sent[:token.i - sent.start])):
                        if prev.ent_type_ in ('PERSON','ORG','GPE'):
                            resolved = prev.text
                            break
                        if prev.pos_ == 'NOUN' and prev.text.lower() in ('woman','man','person','girl','boy'):
                            resolved = prev.text
                            break
                    if not resolved:
                        for prev_sent in doc.sents:
                            for ent in prev_sent.ents:
                                if ent.label_ == "PERSON":
                                    resolved = ent.text
                                    break
                            if resolved: break
                    if resolved:
                        subj = resolved

                # Require subject + (object or prep phrase) to reduce noise
                if subj and (dobj or prep_phrases):
                    events.append({
                        'subject': subj,
                        'verb': token.lemma_,
                        'object': dobj,
                        'prep_phrases': prep_phrases,
                        'negated': negated,
                        'clause': sent.text.strip()
                    })

    # Relative clause patterns
    matcher = Matcher(nlp.vocab)
    pattern_where = [{"LOWER": "where"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN", "OP": "+"}, {"POS": "VERB", "OP": "+"}]
    pattern_who = [{"LOWER": "who"}, {"POS": "VERB", "OP": "+"}]
    matcher.add('RELCLAUSE', [pattern_where, pattern_who])
    for _id, start, end in matcher(doc):
        span = doc[start:end].sent
        descriptions.append(span.text)

    # Flatten event phrases for retrieval
    event_phrases = []
    for ev in events:
        phrase = " ".join(filter(None, [ev['subject'], ev['verb'], ev['object']]))
        if phrase.strip():
            event_phrases.append(phrase.lower())
    descriptions.extend(event_phrases)

    # Clean entities and deduplicate descriptions
    entities = clean_entities(entities, doc)
    descriptions = [d for d in dict.fromkeys(descriptions) if 2 <= len(d.split()) <= 10]

    return {
        "media_type": sorted(set(media_type)),
        "entities": sorted(set(entities)),
        "time": sorted(set(time)),
        "descriptions": sorted(set(descriptions)),
        "events": events
    }

# Example test
# if __name__ == "__main__":
#     q = "George Clooney movie where a woman witnesses a murder through the window and he goes to trial"
#     print(decompose_query(q))