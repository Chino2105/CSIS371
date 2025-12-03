import json
import re

import nltk
from nltk.corpus import stopwords
from pyserini.search.lucene import LuceneSearcher

# import queryDecomposer
from queryDecomposerImproved import decompose_query

# File: searcher.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 12/3/2024
# Description: This program performs search using Pyserini and
# incorporates query decomposition for improved search relevance.

try:
    stop_words = set(stopwords.words('english'))
except OSError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Highlight snippet with query terms in bold (**term**)
def highlight_snippet(contents, query_terms, window=100):
    # Filter out stopwords and very short tokens
    meaningful_terms = [
        t for t in query_terms
        if t.lower() not in stop_words and len(t) > 2
    ]
    if not meaningful_terms:
        # fallback: just return first 150 chars
        return contents[:150].replace("\n", " ")

    # Anchor snippet around first match of a meaningful term
    anchor_idx = None
    for term in meaningful_terms:
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

    # Build one regex for all meaningful terms
    pattern = r'\b(' + '|'.join(re.escape(t) for t in meaningful_terms) + r')\b'
    snippet = re.sub(pattern, r'**\1**', snippet, flags=re.IGNORECASE)

    return snippet


# Construct weighted query based on components
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

       # Normalize like your searcher
        norm_query = input_query.encode("utf-8").decode("unicode_escape")
        norm_query = norm_query.lower()
        norm_query = re.sub(r"[^\w\s]", "", norm_query)
        tokens = [t for t in norm_query.split() if t not in stop_words]
        norm_query = " ".join(tokens)

        # Decompose
        components = decompose_query(input_query)

        # Weighted query
        weighted_query = construct_weighted_query(components, norm_query)
        weighted_hits = searcher.search(weighted_query, k=100)
        results = {"weighted": [(hit.docid, hit.score) for hit in weighted_hits]}

        # Subqueries
        subqueries = [norm_query]
        subqueries.extend(components.get("entities", []))
        subqueries.extend(components.get("time", []))
        subqueries.extend(components.get("descriptions", []))
        subqueries.extend(components.get("media_type", []))
        subqueries.extend(components.get("attributes", []))

        for sq in subqueries:
            hits = searcher.search(sq, k=50)
            results[sq] = [(hit.docid, hit.score) for hit in hits]

        # Fuse
        fused = reciprocal_rank_fusion(results, k=50)
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