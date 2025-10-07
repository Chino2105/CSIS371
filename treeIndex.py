import os
import re
import xml.etree.ElementTree as ET

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/6/2024
# File: treeIndex.py
# Description: This program processes a collection of documents in XML format, extracts text content,
# and makes it searchable by creating a list of unique words.

text = ""                                                # Stores the content of the TEXT tag
uniqueWords = []                                         # Stores unique words found in the TEXT
stopwords = set(nltk.corpus.stopwords.words('english'))  # Set of common stopwords to ignore
lemmatizer = WordNetLemmatizer()                         # Initialize the lemmatizer
directory = "Docs"                                       # Directory containing XML files

# Cleans tokens by removing punctuation and unwanted characters, removing stopwords, and making lowercase
def clean_tokens(token_list):
    cleaned = []
    for word in token_list:
        # Remove punctuation and apostrophe+char at end
        word = re.sub(r"'[A-Za-z]$", '', re.sub(r'[^A-Za-z\']', '', word))
        word = word.lower()  # Normalize case
        if word and word not in stopwords:
            cleaned.append(word)
    return cleaned

# Maps POS tags to WordNet POS tags for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

# Goes through each file in the specified directory
for name in os.listdir(directory):
    filepath = os.path.join(directory, name)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into individual <DOC> blocks
    docs = content.split('<DOC>')
    for doc in docs:
        if not doc.strip():
            continue  # Skip empty chunks

        xml_fragment = '<DOC>' + doc  # Re-add the tag

        # Parses the XML fragment and extracts metadata
        root = ET.fromstring(xml_fragment)
        metadata = {}
        for child in root:
            match child.tag:
                case "TEXT":
                    metadata["TEXT"] = child.text
                case "DOCNO":
                    metadata["DOCNO"] = child.text
    
        tokens = clean_tokens(word_tokenize(metadata.get("TEXT", "")))
        tagged_tokens = nltk.pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tagged_tokens]
        for word in lemmatized:
            if word not in uniqueWords:
                uniqueWords.append(word)

    print (uniqueWords)

        




