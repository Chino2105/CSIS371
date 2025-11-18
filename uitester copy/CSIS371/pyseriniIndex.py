from os import path
from subprocess import run
from sys import executable

# File: pyseriniIndex.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/6/2024
# Description: This program processes a collection of documents in JSONL format, extracts text content,
# and makes it searchable by creating a list of unique words and indexing the documents using Pyserini.
# Results are displayed in TREC format and ranked using the BM25 algorithm.

#--input "TestCorpus" for classical searching, "MasterCorpus" for grading 


index_dir = 'indexes/myindex'                       # Directory to store the index

def index():
    # Check if index already exists
    if not path.exists(index_dir) or not path.exists(path.join(index_dir, "segments_1")):
        run([
        executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",                 
        "--input", "TestCorpus",                              # Input directory with JSONL files
        "--index", "indexes/myindex",                               # Output index directory
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "12",
        "--storeDocvectors",
        ])
        #uncomment for troubleshooting, also set run({ to result =run({ 
       # if result.returncode != 0:
        #   print("Indexing process failed (non-zero exit code).")
         #  return

    # doc index checker
   # from pyserini.search.lucene import LuceneSearcher
   # searcher = LuceneSearcher(index_dir)
   # print(f"Lucene index at '{index_dir}' has {searcher.num_docs} documents.")