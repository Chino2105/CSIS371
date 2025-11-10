from packageInstaller import installPackages
from pyseriniIndex import index
from searcher import search

# File: main.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This is the main program that integrates query decomposition and document indexing. It allows users to input queries,
# decomposes them into structured components using GPT, and searches the indexed documents using Pyserini.

def main():
    # Install Packages
    installPackages()

    # Ensure the index is created
    index()

    # Search
    search()

main()