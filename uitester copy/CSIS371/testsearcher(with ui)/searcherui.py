# File: searcher.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/18/2024
# Description: Interactive shell for querying the Pyserini index with
# query decomposition, history, rerun, and index management commands.



#fix who grading, give more power to unique words, such as wizard look back at wizard queue:

import math
import json
import os
import shutil
from datetime import datetime
from gpt import sendGPT
from pyserini.search.lucene import LuceneSearcher
from queryDecomposerAI import decompose_query
import shutil 

INDEX_DIR = 'indexes/myindex'




RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"



STOPWORDS = {
    "a", "an", "the", "who", "is", "are", "was", "were",
    "with", "about", "to", "for", "from", "in", "on", "of",
    "that", "this", "it", "and", "or", "as", "by", "at",
}





def _clean_tokens(text: str) -> list[str]:
    tokens = []
    for tok in text.lower().split():
        tok = tok.strip()
        if not tok:
            continue
        if tok in STOPWORDS:
            continue
        if len(tok) <= 2:
            continue
        tokens.append(tok)
    return tokens

def build_ai_query(raw_query: str) -> tuple[str, dict]:
    """
    Use the decomposer (spaCy or GPT) to build a better BM25 query string.

    Returns:
        (bm25_query_string, components_dict)

    If decomposition fails or gives nothing useful, we fall back to a cleaned
    version of the raw query.
    """
    components = decompose_query(raw_query)

    media = components.get("media_type", []) or []
    entities = components.get("entities", []) or []
    attrs = components.get("attributes", []) or []   # GPT version; spaCy will just give []
    time = components.get("time", []) or []
    desc = components.get("descriptions", []) or []

    terms: list[str] = []

    # 1) Media type: lightly emphasize it (no triple boost).
    for m in media:
        for tok in _clean_tokens(m):
            terms.append(tok)  # e.g. "movie"

    # 2) Attributes (e.g., "sci-fi", "horror").
    for a in attrs:
        for tok in _clean_tokens(a):
            terms.append(tok)

    # 3) Entities (e.g., "wizard boy", "robot kid").
    for e in entities:
        for tok in _clean_tokens(e):
            terms.append(tok)

    # 4) Time (e.g., "1990s").
    for t in time:
        for tok in _clean_tokens(t):
            terms.append(tok)

    # 5) Descriptions as backoff.
    for d in desc:
        for tok in _clean_tokens(d):
            terms.append(tok)

    # If decomposition gave us nothing, fall back to cleaned raw query.
    if not terms:
        terms = _clean_tokens(raw_query)
        if not terms:
            # absolute worst case, just use raw
            return raw_query, components

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    bm25_query = " ".join(deduped)
    return bm25_query, components


def search():


    


    """
    Interactive Tip-of-the-Tongue shell.

    - Type a normal line → treated as a search query.
    - Type commands starting with ':' for shell features (history, rerun, reindex, etc.).
    """

    def timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")

    # Try to open Lucene index
    if not os.path.exists(INDEX_DIR):
        print(f"[!] Index directory '{INDEX_DIR}' not found.")
        print("    Run main.py so pyseriniIndex.index() can build the index.")
        return

    try:
        searcher = LuceneSearcher(INDEX_DIR)
    except Exception as e:
        print(f"[!] Failed to open Lucene index at '{INDEX_DIR}': {e}")
        return

    query_history = []  # list of raw query strings in order

    print("=" * 70)
    print("Tip-of-the-Tongue Interactive Shell")
    print("Type a query to search, or a command starting with ':'")
    print("")
    print("Examples:")
    print("  that sci-fi movie with the robot kid from the 1990s")
    print("")
    print("Commands:")
    print("  :help                  show help")
    print("  :history               list past queries and their scores")
    print("  :rerun N               rerun query #N from history")
    print("  :clear                 clear query history")
    print("  :reindex               delete + rebuild the Lucene index, then reload")
    print("  :quit / :q / :exit     exit the shell")
    print("  :gpt <request>         use GPT to help refine or suggest queries   ")
    print("=" * 70)

    # -----------------------------
    # Helper: pretty-print results
    # -----------------------------
    def pretty_print_results(query_text, hits):
        print(f"\n[{timestamp()}] Results for query: {query_text!r}")
        if not hits:
            print("[i] No results.")
            return None, []

        # Collect BM25 scores
        scores = [hit.score for hit in hits]

        # ---- small a: per-doc softmax over scores ----
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores) or 1.0   # avoid division by zero
        probs = [e / sum_exp for e in exp_scores]

        # ---- Big A: entropy-based query assurance in [0, 1] ----
        if len(probs) > 1:
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            max_entropy = math.log(len(probs))
            h_norm = entropy / max_entropy
            big_A = 1.0 - h_norm
        else:
            big_A = 1.0  # only one result ⇒ fully concentrated

        print("-" * 80)
        print("Rank | DocID      | Score   | Weighted Score (A/a)")
        print("-" * 80)

        # Use index-based loop so we can grab probs[i] safely
        for i, hit in enumerate(hits):
            rank = i + 1
            a_i = probs[i]                  # small a for this doc
            A_a_str = f"{big_A:0.3f}/{a_i:0.3f}"
            color = color_for_bm25(hit.score)
            print(f"{rank:4d} | {hit.docid:10s} | {color}{hit.score:7.4f}{RESET} | {A_a_str}")

        print("-" * 80)
        return big_A, probs
    # -----------------------------
    # Helper: run a single query
    # -----------------------------
    def run_query(query_text, from_history_id=None):
         # Build AI-shaped BM25 query + get components
        bm25_query, components = build_ai_query(query_text)

        print(f"\n[{timestamp()}] Decomposed query components:")
        for k in ("media_type", "entities", "attributes", "time", "descriptions"):
            print(f"  {k:12s}: {components.get(k, [])}")

        print(f"\n[AI query] Original: {query_text!r}")
        print(f"[AI query] BM25 uses: {bm25_query!r}")

        hits = searcher.search(bm25_query, k=10)
        big_A, probs = pretty_print_results(bm25_query, hits)

        # Save stats in the latest history entry
        if query_history:
            entry = query_history[-1]
            entry["bm25"] = bm25_query
            if big_A is not None and probs:
                entry["big_A"] = big_A
                entry["top_a"] = probs[0]        # a for rank-1 doc
                entry["top_score"] = hits[0].score


    # -----------------------------
    # Commands
    # -----------------------------
    def cmd_help():
        print(
            """
Shell commands (prefix with ':'):

  :help
      Show this help message.

  :history [pattern]
      Show all past queries. If [pattern] is given, only show queries that
      contain that substring (case-insensitive).

  :rerun N
      Rerun query number N from the history list (see :history).

  :clear
      Clear the in-memory query history. Does NOT touch the index.

  :reindex
      Delete the Lucene index directory, rebuild it using pyseriniIndex.index(),
      and then reload it into the searcher. This takes time.
  :gpt <questions>
        Uses GPT for help or suggestions for queries
  :quit / :q / :exit
      Exit the shell.

Anything that does NOT start with ':' is treated as a normal search query.
"""
        )
    def cmd_gpt(arg: str):
        """
         Use the GPT helper.

        - If arg is provided: treat it as the user's question / description.
        - If arg is empty but we have history: use the last query and ask GPT to refine it.
        """
        if not arg:
            if not query_history:
                print("[!] Usage: :gpt <question or description>")
                print("    (No history yet, so I can’t auto-use a last query.)")
                return

            # Default behavior: refine the last query
            last_query = query_history[-1]
            prompt = f"""
You are helping a user improve keyword queries for a BM25 search engine over short English text documents.

The user previously searched for:
"{last_query}"

1. Briefly explain what this query is likely looking for.
2. Suggest ONE "strict" keyword query that uses key terms (fewer words, more specific).
3. Suggest ONE "broad" keyword query (more recall).
4. List 5–10 high-value keywords or short phrases they could try.

Return your answer as plain text, with short bullet points where appropriate.
"""
        else:
            # User provided a custom question for GPT
            prompt = f"""
You are a helper inside an information retrieval shell.
The user says:

\"\"\"{arg}\"\"\"

They are working with a BM25/Lucene-based search over short English text documents.
Help them by:
1. Interpreting what they seem to want.
2. Suggesting one or two good keyword queries for the search engine.
3. Suggesting 5–10 high-value keywords or short phrases.

Return your answer as plain text.
"""

        try:
            answer = sendGPT(prompt)
        except Exception as e:
            print(f"[!] Error calling GPT helper: {e}")
            return

        print("\n[GPT helper]")
        print(answer.strip())
        print()

    def cmd_history(pattern: str):
        if not query_history:
            print("[i] No queries run yet.")
            return

        patt = pattern.strip().lower()
        print("-" * 60)
        print("Query history:")
        for idx, entry in enumerate(query_history, start=1):
            raw = entry["raw"]
            if patt and patt not in raw.lower():
                continue

            A = entry.get("big_A")
            a1 = entry.get("top_a")
            bm25 = entry.get("top_score")

            if A is not None and a1 is not None and bm25 is not None:
                print(
                    f"{idx:3d}: {raw}  "
                    f"[A={A:0.3f}, a1={a1:0.3f}, BM25={bm25:0.4f}]"
                )
            else:
                # Query ran before we added scoring, or had no results
                print(f"{idx:3d}: {raw}")
        print("-" * 60)
    def color_for_bm25(score: float) -> str:
    
        if score >= 3.2:
            return GREEN      # good match
        elif score >= 2.0:
            return YELLOW     # okay / middling match
        else:
            return RED        # weak match

    def cmd_rerun(arg: str):
        if not arg:
            print("[!] Usage: :rerun N")
            return
        try:
            n = int(arg)
        except ValueError:
            print("[!] N must be an integer.")
            return

        if n < 1 or n > len(query_history):
            print(f"[!] No query #{n} in history.")
            return

        entry = query_history[n - 1]
        q = entry["raw"]
        print(f"[{timestamp()}] Rerunning query #{n}: {q!r}")
        run_query(q, from_history_id=n)
    
    def cmd_clear():
        query_history.clear()
        print("[i] Cleared query history (index is untouched).")

    def cmd_reindex():
        nonlocal searcher
        from pyseriniIndex import index as build_index

        confirm = input(
            "[?] This will DELETE and rebuild the index at "
            f"'{INDEX_DIR}'. Continue? (y/N): "
        ).strip().lower()
        if confirm != "y":
            print("[i] Reindex cancelled.")
            return

        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
            print(f"[{timestamp()}] Deleted old index directory '{INDEX_DIR}'.")

        print(f"[{timestamp()}] Rebuilding index...")
        build_index()
        print(f"[{timestamp()}] Reloading LuceneSearcher...")
        searcher = LuceneSearcher(INDEX_DIR)
        print(f"[{timestamp()}] Reindex complete.")

    def handle_command(rest: str):
        parts = rest.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("help", "h", "?"):
            cmd_help()
        elif cmd == "history":
            cmd_history(arg)
        elif cmd == "rerun":
            cmd_rerun(arg)
        elif cmd == "clear":
            cmd_clear()
        elif cmd in ("reindex", "rebuild"):
            cmd_reindex()
        elif cmd in ("gpt", "ai", "assistant"):
            cmd_gpt(arg)
        elif cmd in ("quit", "q", "exit"):
            # Use SystemExit so we can cleanly break in the main loop.
            raise SystemExit
        else:
            print(f"[!] Unknown command ':{cmd}'. Type ':help' for options.")

    # -----------------------------
    # Main REPL loop
    # -----------------------------
    while True:
        try:
            line = input("ToT: ").strip()
        except KeyboardInterrupt:
            # Don't immediately exit on Ctrl+C; nudge user to use :quit
            print("\n[i] Press Ctrl+D or type ':quit' to exit.")
            continue
        except EOFError:
            # Ctrl+D
            print("\n[+] Goodbye.")
            break

        if not line:
            continue

        # Commands start with ':'
        if line.startswith(":"):
            try:
                handle_command(line[1:])
            except SystemExit:
                print("[+] Goodbye.")
                break
            # IMPORTANT: do NOT fall through to run_query
            continue

        # Otherwise: normal search query
        query_history.append({
            "raw": line,
            "bm25": None,
            "big_A": None,
            "top_a": None,
            "top_score": None,
        })
        run_query(line)
