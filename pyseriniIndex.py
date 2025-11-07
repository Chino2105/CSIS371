import os
import subprocess

# File: pyseriniIndex.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/29/2024
# Description: This program processes a collection of documents in JSONL format, extracts text content,
# and makes it searchable by creating a list of unique words and indexing the documents using Pyserini.
# Results are displayed in TREC format and ranked using the BM25 algorithm.

index_dir = 'indexes/myindex'                       # Directory to store the index

if __name__ == "__main__":

    # Check if index already exists
    if not os.path.exists(index_dir) or not os.path.exists(os.path.join(index_dir, "segments_1")):
        subprocess.run([
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",                 
        "--input", "CORPUS_converted",                              # Input directory with JSONL files
        "--index", "indexes/myindex",                               # Output index directory
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8"
        ])

        
    