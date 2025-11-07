from pyserini.search.lucene import LuceneSearcher

import queryDecomposer

# Interactive search loop
searcher = LuceneSearcher('indexes/myindex')

input_query = input("Search query: ").strip()
while input_query != "":
    # Decompose the query
    components = queryDecomposer.decompose_query(input_query)
    print("Decomposed Query Components:", components)

    # Search the index
    hits = searcher.search(input_query, k=10)
    if not hits:
        print("No results found.")
    for i, hit in enumerate(hits):
        print(f"0 1 {hit.docid} {i+1} {hit.score:.4f} baseline")
    input_query = input("\nSearch query: ").strip()