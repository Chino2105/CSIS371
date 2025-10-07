import os
import re
import xml.etree.ElementTree as ET

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')


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
    
class Node:
    def __init__(self, keys=None, children=None):
        self.keys = keys or []
        self.children = children or []

    def is_leaf(self):
        return len(self.children) == 0

    def insert_key(self, key):
        self.keys.append(key)
        self.keys.sort()

class TwoThreeTree:
    def __init__(self):
        self.root = None

    def search(self, node, key):
        if node is None:
            return False
        if key in node.keys:
            return True
        if node.is_leaf():
            return False
        if key < node.keys[0]:
            return self.search(node.children[0], key)
        elif len(node.keys) == 1 or key < node.keys[1]:
            return self.search(node.children[1], key)
        else:
                return self.search(node.children[2], key)

    def insert(self, key):
        if self.root is None:
            self.root = Node([key])
        else:
            self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node.is_leaf():
            node.insert_key(key)
            if len(node.keys) <= 2:
                return node
            return self._split(node)
        else:
            if key < node.keys[0]:
                idx = 0
            elif len(node.keys) == 1 or key < node.keys[1]:
                idx = 1
            else:
                idx = 2

            child = self._insert(node.children[idx], key)
            node.children[idx] = child

            if len(child.keys) == 3:
                return self._split_internal(node, idx)
            return node

    def _split(self, node):
        left = Node([node.keys[0]])
        right = Node([node.keys[2]])
        return Node([node.keys[1]], [left, right])

    def _split_internal(self, parent, child_index):
        child = parent.children[child_index]
        left = Node([child.keys[0]])
        right = Node([child.keys[2]])
        middle_key = child.keys[1]

        if child.is_leaf():
            left.children = []
            right.children = []
        else:
            left.children = child.children[:2]
            right.children = child.children[2:]

        parent.keys.insert(child_index, middle_key)
        parent.children[child_index:child_index+1] = [left, right]

        if len(parent.keys) > 2:
            return self._split(parent)
        return parent

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

tree = TwoThreeTree()
for word in uniqueWords:
    tree.insert(word) 

        




