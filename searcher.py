import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pyserini.search.lucene import LuceneSearcher

# import queryDecomposer
from queryDecomposerImproved import decompose_query

# File: searcher.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/19/2024
# Description: This program performs search using Pyserini and
# incorporates query decomposition for improved search relevance.

try:
    stop_words = set(stopwords.words('english'))
except OSError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Highlight snippet with query terms in bold (**term**)
def highlight_snippet(contents, query_terms, window=100):
    # Anchor snippet around first match
    anchor_idx = None
    for term in query_terms:
        match = re.search(r'\b' + re.escape(term) + r'\b', contents, re.IGNORECASE)
        if match:
            anchor_idx = match.start()
            break

    if anchor_idx is not None:
        start = max(0, anchor_idx - 50)
        end = min(len(contents), anchor_idx + window)
        snippet = contents[start:end].replace("\n", " ")
    else:
        snippet = contents[:150].replace("\n", " ")

    # Build one regex for all terms
    pattern = r'\b(' + '|'.join(re.escape(t) for t in query_terms) + r')\b'
    snippet = re.sub(pattern, r'**\1**', snippet, flags=re.IGNORECASE)

    return snippet

# Reciprocal Rank Fusion implementation
def reciprocal_rank_fusion(results, k=20, c=60):
    scores = {}
    for sq, hits in results.items():
        for rank, (docid, _) in enumerate(hits):
            scores[docid] = scores.get(docid, 0) + 1.0 / (c + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

# Main search function
def search():
    # Interactive search loop
    searcher = LuceneSearcher('indexes/myindex')
    searcher.set_bm25(k1=1.2, b=0.75)

    input_query = input("Search query: ").strip()
    while input_query != "":

        print()

        # Preprocess the query
        input_query = input_query.encode('utf-8').decode('unicode_escape')
        input_query = input_query.lower()
        input_query = re.sub(r'[^\w\s]', '', input_query)

        tokens = input_query.split()
        tokens = [t for t in tokens if t not in stop_words]
        #stemmer = PorterStemmer()
        #tokens = [stemmer.stem(t) for t in tokens]

        input_query = " ".join(tokens)

        # Decompose the query
        components = decompose_query(input_query)
        #print("Decomposed Query Components:", components)

        # Perform Reciprocal Rank Fusion
        subqueries = []
        subqueries.append(input_query)
        for entity in components.get('entities', []):
            subqueries.append(entity)
        for time in components.get('time', []):
            subqueries.append(time)
        for desc in components.get('descriptions', []):
            subqueries.append(desc)
        for media in components.get('media_type', []):
            subqueries.append(media)

        # Search each subquery and collect results
        results = {}
        for sq in subqueries:
            hits = searcher.search(sq, k=50)
            results[sq] = [(hit.docid, hit.score) for hit in hits]

        # Fuse results using Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(results, k=50, c=100)
        for i, (docid, score) in enumerate(fused):
            doc = searcher.doc(docid)
            if doc is None:
                continue
            raw = json.loads(doc.raw())
            contents = raw.get("contents", "")

            # Title = first line before newline
            title = contents.split("\n", 1)[0]

            # Snippet = find first query term in contents
            snippet = highlight_snippet(contents, input_query.split())

            print(f"Rank {i+1} | DocID: {docid} | Score: {score:.4f}")
            print(f"Title: {title}")
            print(f"Snippet: {snippet}\n")

        input_query = input("\nSearch query: ").strip()