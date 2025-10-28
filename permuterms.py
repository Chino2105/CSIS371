import re

# permuterms.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/6/2024
# Description: This module implements a permuterm index to support wildcard searches.

permuterm_index = {}
original_terms = []

def generate_permuterms(terms):
    global permuterm_index, original_terms
    permuterm_index.clear()
    original_terms.clear()

    for term in terms:          # iterate over each string
        term = term + "$"       # append end marker
        original_terms.append(term[:-1])  # store original without $
        for i in range(len(term)):
            rotated = term[i:] + term[:i]
            permuterm_index[rotated] = term[:-1]

def wildcard_to_regex(query):
    escaped = re.escape(query)
    regex = "^" + escaped.replace("\\*", ".*") + "$"
    return regex

def search_permuterm(query, permuterm_index, original_terms):
    # e.g. query = "pol*"
    if "*" not in query:
        return [query] if query in original_terms else []

    prefix, suffix = query.split("*", 1)
    rotated_query = suffix + "$" + prefix

    results = []
    for rotated, original in permuterm_index.items():
        if rotated.startswith(rotated_query):
            results.append(original)
    return results

