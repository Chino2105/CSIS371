from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.search.hybrid import HybridSearcher
import json

# ==========================================
# 1. CONFIGURATION & LOADING
# ==========================================
# Path to the text index (BM25)
sparse_path = '/mnt/data/indexes/sparse_index'
# Path to the vector index (FAISS)
dense_path = '/mnt/data/indexes/dense_embeddings'

print("Loading Sparse Index (Text)...")
try:
    s_searcher = LuceneSearcher(sparse_path)
except Exception as e:
    print(f"CRASH: Could not load Sparse Index at {sparse_path}")
    print("Did you run the 'pyserini.index.lucene' command?")
    exit(1)

print("Loading Dense Index (Vectors)...")
try:
    # We use the same model to encode the query
    d_searcher = FaissSearcher(
        dense_path,
        'sentence-transformers/all-MiniLM-L6-v2'
    )
except Exception as e:
    print(f"CRASH: Could not load Dense Index at {dense_path}")
    exit(1)

# Initialize Hybrid Searcher (Combines both)
h_searcher = HybridSearcher(d_searcher, s_searcher)

# ==========================================
# 2. DEFINE QUERY
# ==========================================
query = "ENGLISH MOVIE MEN DISGUISE AS WOMAN AND LIVE WITH WOMAN. I dont remember most of this scene in this movie. I watch this movie about 2004-2005. the movie is about 3 men whose lost their money and they were suspecting a group of women. So, in order to investigate the women, they disguise themselve as woman and make excuse to live with the women. The scene that remember the most is where a man taking bath at night and suddenly a woman joined him. But the girl didnt realizes that the man is a man caused she didnt wear her glasses."

# ==========================================
# 3. EXECUTE SEARCH
# ==========================================
print(f"\nSearching for: {query[:50]}...")

# Option A: Pure Dense Search (What your original code did)
# hits = d_searcher.search(query, k=10)

# Option B: Hybrid Search (Better Results)
# alpha=0.1 -> 90% Vector Score + 10% Keyword Score
hits = h_searcher.search(query, k=10, alpha=0.1)

# ==========================================
# 4. DISPLAY RESULTS
# ==========================================
print("\n" + "=" * 60)
print(f"{'RANK':<5} | {'SCORE':<8} | {'TITLE / SNIPPET'}")
print("=" * 60)

for i, hit in enumerate(hits):
    doc_id = hit.docid
    score = hit.score

    # FETCH TEXT: This is why we needed Step 1
    # We ask the Sparse Searcher: "Give me the raw text for this ID"
    try:
        raw_doc = s_searcher.doc(doc_id).raw()
        content_json = json.loads(raw_doc)

        # Handle cases where field might be 'text' or 'contents'
        text_content = content_json.get('text') or content_json.get('contents') or "No Text Found"

        # Print a snippet
        print(f"{i + 1:<5} | {score:.4f}   | {text_content[:100]}...")
        print(f"      (ID: {doc_id})")
        print("-" * 60)

    except Exception as e:
        print(f"Error reading doc {doc_id}: {e}")