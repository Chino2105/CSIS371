import json
import os
import re

from pyserini.search.lucene import LuceneSearcher

from queryDecomposer import decompose_query
from searcher import (construct_weighted_query, reciprocal_rank_fusion,
                      stop_words)


def load_queries(jsonl_path):
    queries = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["query_id"]] = obj["query"]
    return queries

def load_results(results_path):
    expected = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rank = parts[:4]
                expected[qid] = {"docid": docid, "rank": int(rank)}
    return expected

def run_test(queries, expected, index_dir="indexes/myindex"):
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=1.2, b=0.75)

    total = 0
    correct = 0

    for qid, query in queries.items():
        print(f"\n=== Testing Query {qid} ===")
        # Normalize like your searcher
        norm_query = query.encode('utf-8').decode('unicode_escape')
        norm_query = norm_query.lower()
        norm_query = re.sub(r'[^\w\s]', '', norm_query)
        tokens = [t for t in norm_query.split() if t not in stop_words]
        norm_query = " ".join(tokens)

        # Decompose
        components = decompose_query(query)
        #print("Decomposed:", components)

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

        for sq in subqueries:
            hits = searcher.search(sq, k=50)
            results[sq] = [(hit.docid, hit.score) for hit in hits]

        # Fuse
        fused = reciprocal_rank_fusion(results, k=50)
        # for i, (docid, score) in enumerate(fused):
        #     print(f"0 Q0 {docid} {i+1} {score:.4f} fused")

        # Check expected
        exp = expected.get(qid)
        if exp:
            total += 1
            found = next((i for i, (docid, _) in enumerate(fused) if docid == exp["docid"]), None)
            if found is not None:
                correct += 1
                print(f"Expected doc {exp['docid']} found at fused rank {found+1}")
            else:
                print(f"Expected doc {exp['docid']} not in fused top results")
        else:
            print("No expected result for this query")

    # Final accuracy report
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n=== Summary ===")
        print(f"Correct: {correct}/{total} queries")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo queries with expected results to evaluate.")

def run_all_tests(query_folder, result_folder):
    query_files = sorted(os.listdir(query_folder))
    result_files = sorted(os.listdir(result_folder))
    for qf, rf in zip(query_files, result_files):
        print(f"\n=== Running test set: {qf} with {rf} ===")
        queries = load_queries(os.path.join(query_folder, qf))
        expected = load_results(os.path.join(result_folder, rf))
        run_test(queries, expected)

if __name__ == "__main__":
    run_all_tests("Test Queries", "Test Query Results")