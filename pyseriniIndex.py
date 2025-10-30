import os
import subprocess

from pyserini.search.lucene import LuceneSearcher

# File: pyseriniIndex.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/29/2024
# Description: This program processes a collection of documents in JSONL format, extracts text content,
# and makes it searchable by creating a list of unique words and indexing the documents using Pyserini.
# Results are displayed in TREC format and ranked using the BM25 algorithm.

index_dir = 'indexes/myindex'                       # Directory to store the index

#__import__('os').system('pip install pyserini')

if __name__ == "__main__":

    # Check if index already exists
    if not os.path.exists(index_dir) or not os.path.exists(os.path.join(index_dir, "segments_1")):
        subprocess.run([
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", "TestCorpus",
        "--index", "indexes/myindex",
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8"
        ])


    # Create a searcher to search index
    searcher = LuceneSearcher('indexes/myindex')
    searcher.set_bm25(k1=0.9, b=0.4)

    # Interactive search loop
    input_query = input("Search query: ").strip()
    while input_query != "":
        hits = searcher.search(input_query, k=10)
        if not hits:
            print("No results found.")
        for i, hit in enumerate(hits):
            print(f"0 1 {hit.docid} {i+1} {hit.score:.4f} baseline")
        input_query = input("\nSearch query: ").strip()
        
    