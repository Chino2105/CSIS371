import json
from tqdm import tqdm

input_path = '../CORPUS_converted'      # Your current file (id, contents)
output_path = '../CORPUS_for_encoder.jsonl' # New file (id, text)

print("Renaming 'contents' -> 'text'...")

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in tqdm(fin):
        try:
            data = json.loads(line)
            # We keep the combined Title + Plot, but rename the key to "text"
            new_data = {
                "id": data["id"],
                "text": data["contents"]
            }
            fout.write(json.dumps(new_data) + "\n")
        except:
            continue