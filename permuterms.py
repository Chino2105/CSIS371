# permuterms.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/6/2024
# Description: This module implements a permuterm index to support wildcard searches.

def generate_permuterms(term):
    term = term + "$"
    return [term[i:] + term[:i] for i in range(len(term))]

# Example index
permuterm_index = {}
terms = ["cat", "bat", "rat", "car"]
for term in terms:
    for rotation in generate_permuterms(term):
        permuterm_index[rotation] = term

def wildcard_to_prefix(query):
    if "*" not in query:
        return query + "$"
    parts = query.split("*")
    return parts[1] + "$" + parts[0]

def search_permuterm(query, index):
    prefix = wildcard_to_prefix(query)
    matches = [index[rotation] for rotation in index if rotation.startswith(prefix)]
    return list(set(matches))  # remove duplicates
