import json
import os

# File: corpusFormatConverter.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 10/29/2024
# Description: This program converts a corpus of JSONL documents into a format compatible with Pyserini.

input_dir = "CORPUS_split"
output_dir = "CORPUS_converted"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".jsonl"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

            doc_id = str(obj.get("id", "")).strip()
            url = obj.get("url", "")
            title = obj.get("title", "")
            text = obj.get("text", "")

            if not doc_id or not text:
                continue  # skip if missing essentials

            # Concatenate fields into contents
            contents = f"{title}\n{url}\n{text}"

            new_obj = {
                "id": doc_id,
                "contents": contents
            }
            outfile.write(json.dumps(new_obj) + "\n")

    print(f"Converted {filename} -> {output_path}")

print("Conversion complete. Use CORPUS_converted/ for Pyserini indexing.")