import re

# permuterms.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/6/2024
# Description: This module implements a permuterm index to support wildcard searches.

def generate_permuterms(term):
    term = term + "$"
    return [term[i:] + term[:i] for i in range(len(term))]

permuterm_index = {}
original_terms = set()

def wildcard_to_regex(query):
    escaped = re.escape(query)
    regex = "^" + escaped.replace("\\*", ".*") + "$"
    return regex

def search_permuterm(query, index, original_terms):
    pattern = re.compile(wildcard_to_regex(query), re.IGNORECASE)
    matches = [term for term in original_terms if pattern.match(term)]
    return matches

