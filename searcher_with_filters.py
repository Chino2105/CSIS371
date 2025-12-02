# Searcher with component filtering and attribute integration
import re

import nltk
from nltk.corpus import stopwords
from pyserini.search.lucene import LuceneSearcher

from queryDecomposer import decompose_query

# File: searcher_with_filters.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 12/02/2025
# Description: Performs search using Pyserini with query decomposition,
# component filtering, attribute integration, and tuned fusion.

try:
    stop_words = set(stopwords.words('english'))
except OSError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# --- Component Filtering ---
def filter_components(components):
    generic_entities = {"man","woman","lady","ghost","dads","people","person"}
    # Filter entities
    components["entities"] = [
        e for e in components.get("entities", [])
        if e and e.lower() not in generic_entities
    ]
    # Normalize attributes
    attrs = []
    for a in components.get("attributes", []):
        a = a.lower()
        if "r rated" in a:
            attrs.append("R-rated")
        elif "crazy shit" in a:
            continue
        else:
            attrs.append(a)
    components["attributes"] = attrs
    # Clean descriptions
    components["descriptions"] = [
        d for d in components.get("descriptions", [])
        if d and len(d.split()) > 1 and "shit" not in d.lower()
    ]
    return components

# --- Weighted Query Construction ---
def construct_weighted_query(components, original_query):
    query_parts = [original_query]
    # Boost entities
    for entity in components.get('entities', []):
        query_parts.append(f'"{entity}"^4')
    # Boost time
    for t in components.get('time', []):
        query_parts.append(f'"{t}"^2')
    # Boost attributes
    for attr in components.get('attributes', []):
        if attr.lower() in {"cannes","awards","art house flick","horror"}:
            query_parts.append(f'"{attr}"^3')
        else:
            query_parts.append(f'"{attr}"^2')
    return " ".join(query_parts)

# --- Reciprocal Rank Fusion ---
def reciprocal_rank_fusion(results, k=20, c=100):
    scores = {}
    for sq, hits in results.items():
        for rank, (docid, _) in enumerate(hits):
            scores[docid] = scores.get(docid, 0) + 1.0 / (c + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

# --- Interactive Search Loop ---
def search():
    searcher = LuceneSearcher('indexes/myindex')
    searcher.set_bm25(k1=1.2, b=0.75)

    input_query = input("Search query: ").strip()
    while input_query != "":
        # Normalize query
        original_query = input_query.encode('utf-8').decode('unicode_escape')
        norm_query = original_query.lower()
        norm_query = re.sub(r'[^\w\s]', '', norm_query)
        tokens = [t for t in norm_query.split() if t not in stop_words]
        norm_query = " ".join(tokens)

        # Decompose and filter
        components = decompose_query(original_query)
        components = filter_components(components)
        print("Filtered Components:", components)

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
        fused = reciprocal_rank_fusion(results, k=50, c=100)
        for i, (docid, score) in enumerate(fused):
            print(f"0 Q0 {docid} {i+1} {score:.4f} fused")

        input_query = input("\nSearch query: ").strip()