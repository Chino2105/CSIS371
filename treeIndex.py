import json
import math
import os
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from nltk.corpus import wordnet

from permuterms import (generate_permuterms, original_terms, permuterm_index,
                        search_permuterm)

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')


# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/27/2024
# File: treeIndex.py
# Description: This program processes a collection of documents in XML format, extracts text content,
# and makes it searchable by creating a list of unique words.

directory = "TestCorpus"                                     # Directory containing miniCorpus JSONL files

# Processes a single JSONL file and returns local index and document vectors
def process_file(filepath):
    print(f"Processing {filepath} ...", flush=True)
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stopwords_set = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    local_index = {}
    local_vectors = defaultdict(dict)

    for obj, line_no in stream_jsonl(filepath):
        doc_id = next((str(obj[k]) for k in ('DOCNO', 'docno', 'id', 'doc_id') if k in obj and obj[k] is not None), f"{os.path.basename(filepath)}:{line_no}")
        text_field = next((obj[k] for k in ('TEXT', 'text', 'content', 'body') if k in obj and obj[k] is not None), None)
        if not text_field:
            continue

        tokens = clean_tokens(word_tokenize(str(text_field)), stopwords_set)
        tagged = nltk.pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(tok, get_wordnet_pos(pos)) for tok, pos in tagged]

        for word in lemmatized:
            if not word:
                continue
            if word not in local_index:
                local_index[word] = Posting(word)
            posting = local_index[word]
            if doc_id not in posting.doc_id:
                posting.doc_id.append(doc_id)
                posting.term_frequency[doc_id] = 1
            else:
                posting.term_frequency[doc_id] += 1
            tf = posting.term_frequency[doc_id]
            log_tf = 1 + math.log10(tf)
            posting.log_term_frequency[doc_id] = log_tf
            local_vectors[doc_id][word] = log_tf

    print(f"Finished {filepath}", flush=True)
    return local_index, local_vectors

# Processes files in parallel and merges local indices into a global index
def parallel_index(directory):
    futures = []
    with ProcessPoolExecutor() as executor:
        for name in os.listdir(directory):
            if name.startswith('miniCorpus_') and name.endswith('.jsonl'):
                filepath = os.path.join(directory, name)
                futures.append(executor.submit(process_file, filepath))

        merged_index = {}
        merged_vectors = defaultdict(dict)

        for future in as_completed(futures):
            local_index, local_vectors = future.result()
            for term, posting in local_index.items():
                if term not in merged_index:
                    merged_index[term] = posting
                else:
                    merged_posting = merged_index[term]
                    for doc_id in posting.doc_id:
                        if doc_id not in merged_posting.doc_id:
                            merged_posting.doc_id.append(doc_id)
                            merged_posting.term_frequency[doc_id] = posting.term_frequency[doc_id]
                        else:
                            merged_posting.term_frequency[doc_id] += posting.term_frequency[doc_id]
                        tf = merged_posting.term_frequency[doc_id]
                        merged_posting.log_term_frequency[doc_id] = 1 + math.log10(tf)

            for doc_id, vec in local_vectors.items():
                for term, weight in vec.items():
                    merged_vectors[doc_id][term] = merged_vectors[doc_id].get(term, 0) + weight

    return merged_index, merged_vectors

# Streams JSONL file line by line, yielding each JSON object
def stream_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj, line_no
            except json.JSONDecodeError:
                continue  # Skip malformed lines

# Normalizes document vectors to unit length
def normalize_vectors(index, vectors):
    for doc_id, term_weights in vectors.items():
        norm = math.sqrt(sum(w ** 2 for w in term_weights.values()))
        if norm == 0:
            continue
        for term, weight in term_weights.items():
            index[term].log_term_frequency[doc_id] = weight / norm

# Cleans tokens by removing punctuation and unwanted characters, removing stopwords, and making lowercase
def clean_tokens(token_list, stopwords):
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
    def insert_entry(self, term, posting):
        for i, (t, plist) in enumerate(self.entries):
            if t == term:
                plist.append(posting)
                return
            
        self.entries.append((term, posting))
        self.entries.sort(key=lambda x: x[0])


class TwoThreeTree:

    # Initializes an empty 2-3 tree
    def __init__(self, index):
        self.root = None
        self.index = index

    # Inserts a term into the tree
    def insert(self, term):
        if self.root is None:
            self.root = Node([(term, self.index[term])])
        else:
            self.root = self._insert(self.root, term, self.index[term])

    # Inserts a term into the tree and returns the new root if the tree was split
    def _insert(self, node, term, doc_id):
        if node.is_leaf():
            node.insert_entry(term, self.index[term])
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
            results = search_permuterm(term, permuterm_index, original_terms)
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
    def __init__(self, term):
        self.term = term
        self.doc_id = []
        self.term_frequency = {}
        self.log_term_frequency = {}

    def __str__(self):
        # Show doc IDs and frequencies
        return f"docs={self.doc_id}, tf={self.term_frequency}"


# Add a document-level dictionary to store document vectors
document_vectors = defaultdict(dict)  # Format: {doc_id: {term: log_tf}}

if __name__ == "__main__":
    from nltk.corpus import stopwords
    query_stopwords = set(stopwords.words('english'))
                      
    # Check if index already exists
    if os.path.exists("index.pkl") and os.path.exists("vectors.pkl"):
        print("Loading existing index from disk...")
        with open("index.pkl", "rb") as f:
            uniqueWords = pickle.load(f)
        with open("vectors.pkl", "rb") as f:
            document_vectors = pickle.load(f)

    # Step 1: Index the corpus
    else:
        print("Indexing corpus in parallel...")
        uniqueWords, document_vectors = parallel_index(directory)
        normalize_vectors(uniqueWords, document_vectors)

        # Save for next time
        with open("index.pkl", "wb") as f:
            pickle.dump(uniqueWords, f)
        with open("vectors.pkl", "wb") as f:
            pickle.dump(document_vectors, f)

    print(f"Indexed {len(uniqueWords)} unique terms across {len(document_vectors)} documents.")

    # Save index and vectors
    with open("index.pkl", "wb") as f:
        pickle.dump(uniqueWords, f)

    with open("vectors.pkl", "wb") as f:
        pickle.dump(document_vectors, f)

    print("Index serialized to disk.")

    print("Normalization complete.")

    # Step 2: Build the 2-3 tree (optional if you still want it)
    # Build permuterm index for wildcard queries
    generate_permuterms(list(uniqueWords.keys()))
    tree = TwoThreeTree(uniqueWords)
    for word in uniqueWords:
        tree.insert(word)

    print("2-3 Tree built. Ready for queries.")

    from nltk.tokenize import word_tokenize

    # Step 3: Query loop
    while True:
        input_query = input("Search query: ").strip()
        if not input_query:
            break
        terms = clean_tokens(word_tokenize(input_query), query_stopwords)
        for term in input_query.split():
            result = tree.search(tree.root, term)
            print(result if result else f"{term} not found.")
