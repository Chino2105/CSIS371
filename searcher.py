import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pyserini.search.lucene import LuceneSearcher

# import queryDecomposer
from queryDecomposer import decompose_query

# File: searcher.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/19/2024
# Description: This program performs search using Pyserini and
# incorporates query decomposition for improved search relevance.

def construct_weighted_query(components, original_query):
    # Start with the original query as a baseline
    query_parts = [original_query]

    # Boost entities heavily (e.g., ^4 means 4x importance)
    for entity in components.get('entities', []):
        query_parts.append(f'"{entity}"^4')

    # Boost time slightly
    for time in components.get('time', []):
        query_parts.append(f'"{time}"^2')

    return " ".join(query_parts)


def search():
    # Interactive search loop
    searcher = LuceneSearcher('indexes/myindex')
    searcher.set_bm25(k1=1.2, b=0.75)

    input_query = input("Search query: ").strip()
    while input_query != "":
        stop_words = set(stopwords.words('english'))
        input_query = input_query.lower()
        input_query = re.sub(r'[^\w\s]', '', input_query)

        tokens = input_query.split()
        tokens = [t for t in tokens if t not in stop_words]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

        input_query = " ".join(tokens)

        # Decompose the query
        components = decompose_query(input_query)
        print("Decomposed Query Components:", components)

        # Search the index
        weighted_query = construct_weighted_query(components, input_query)
        hits = searcher.search(weighted_query, k=100)  # Retrieve more results to rerank later

        if not hits:
            print("No results found.")
        for i, hit in enumerate(hits):
            print(f"0 1 {hit.docid} {i+1} {hit.score:.4f} baseline")
        input_query = input("\nSearch query: ").strip()