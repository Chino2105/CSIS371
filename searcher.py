from pyserini.search.lucene import LuceneSearcher

# import queryDecomposer
from queryDecomposer import decompose_query

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

    input_query = input("Search query: ").strip()
    while input_query != "":
        # Decompose the query
        components = decompose_query(input_query)
        print("Decomposed Query Components:", components)

        # Search the index
        # Inside your search loop:
        components = decompose_query(input_query)
        weighted_query = construct_weighted_query(components, input_query)
        hits = searcher.search(weighted_query, k=100)  # Retrieve more results to rerank later

        if not hits:
            print("No results found.")
        for i, hit in enumerate(hits):
            print(f"0 1 {hit.docid} {i+1} {hit.score:.4f} baseline")
        input_query = input("\nSearch query: ").strip()