import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from permuterms import generate_permuterms, permuterm_index, search_permuterm

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
uniqueWords = {}                                         # Stores unique words found in the TEXT
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
    # Initializes a node with entries and children
    def __init__(self, entries=None, children=None):
        self.entries = entries or []  
        self.children = children or []

    # Checks if the node is a leaf
    def is_leaf(self):
        return len(self.children) == 0

    # Inserts an entry into the node
    def insert_entry(self, term, doc_id):
        for i, (t, plist) in enumerate(self.entries):
            if t == term:
                plist.append(uniqueWords[term])
                return
            
        self.entries.append((term, uniqueWords[term]))
        self.entries.sort(key=lambda x: x[0])


class TwoThreeTree:

    # Initializes an empty 2-3 tree
    def __init__(self):
        self.root = None

    # Inserts a term into the tree
    def insert(self, term):
        if self.root is None:
            self.root = Node([(term, uniqueWords[term])])
        else:
            self.root = self._insert(self.root, term, uniqueWords[term])

    # Inserts a term into the tree and returns the new root if the tree was split
    def _insert(self, node, term, doc_id):
        if node.is_leaf():
            node.insert_entry(term, doc_id)
            if len(node.entries) <= 2:
                return node
            return self._split(node)
        else:
            # Find correct child
            if term < node.entries[0][0]:
                idx = 0
            elif len(node.entries) == 1 or term < node.entries[1][0]:
                idx = 1
            else:
                idx = 2

            child = self._insert(node.children[idx], term, doc_id)
            node.children[idx] = child
            if len(child.entries) > 2:
                return self._split_internal(node, idx)
            return node
        
    # Splits a leaf node
    def _split(self, node):
        left = Node([node.entries[0]])
        right = Node([node.entries[2]])
        return Node([node.entries[1]], [left, right])

    # Splits an internal node
    def _split_internal(self, parent, child_index):
        child = parent.children[child_index]
        left = Node([child.entries[0]])
        right = Node([child.entries[2]])
        middle_entry = child.entries[1]

        left.children = child.children[:2]
        right.children = child.children[2:]

        parent.entries.insert(child_index, middle_entry)
        parent.children[child_index:child_index+1] = [left, right]

        if len(parent.entries) > 2:
            return self._split(parent)
        return parent

    # Searches for a term in the tree and returns its posting list
    def search(self, node, term):
        # Base case: empty node
        if node is None:
            return None
        
        # Handle wildcard search
        if "*" in term:
            results = search_permuterm(term, permuterm_index)
            results_str = ""
            for res in results:
                results_str += self.search(self.root, res) + "\n"
            return results_str.strip()
        
        # Search for the term in the current node
        for i, (t, plist) in enumerate(node.entries):
            if t == term:
                return term + ": " + plist.__str__()
            
        if node.is_leaf():
            return None
        if term < node.entries[0][0]:
            return self.search(node.children[0], term)
        elif len(node.entries) == 1 or term < node.entries[1][0]:
            return self.search(node.children[1], term)
        else:
            return self.search(node.children[2], term)
        
# Class for postings in the posting list
class Posting:
    def __init__(self, doc_id):
        self.doc_id = doc_id

    def __str__(self):
        return str(self.doc_id)
    
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
    
        # Tokenizes, cleans, and lemmatizes the text
        tokens = clean_tokens(word_tokenize(metadata.get("TEXT", "")))
        tagged_tokens = nltk.pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tagged_tokens]
        for word in lemmatized:
            doc_id = metadata.get("DOCNO", "")
            posting = str(Posting(doc_id))
            if word not in uniqueWords:
                uniqueWords[word] = [posting]  # [word, postingList]
                for rotation in generate_permuterms(word):
                    permuterm_index[rotation] = word
            else:
                uniqueWords[word].append(posting)

tree = TwoThreeTree()
for word in uniqueWords:
    posting_list = uniqueWords[word]
    tree.insert(word) 

input_query = input("Search query: ").strip()
while input_query != "":
    print(tree.search(tree.root, input_query))
    input_query = input("Search query: ").strip()